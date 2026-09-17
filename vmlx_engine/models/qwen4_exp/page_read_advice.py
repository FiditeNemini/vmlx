"""Bounded Darwin read hints for nonresident pages of exact PLE selections.

No tensor library import, worker, owned descriptor, or data read lives here.
The caller owns the opt-in policy and must still perform its original reads.
"""

import operator
import os
import sys


_MAX_PAGES = 512
_MAX_BYTES = 8 * 1024 * 1024


class _DarwinBackend:
    def __init__(self):
        # Load platform interfaces only after the Darwin admission check.
        import ctypes
        import fcntl

        class ReadAdvisory(ctypes.Structure):
            _fields_ = [("offset", ctypes.c_int64), ("count", ctypes.c_int32)]

        if ctypes.sizeof(ReadAdvisory) != 16 or ReadAdvisory.count.offset != 8:
            raise ValueError("unsupported radvisory ABI")
        self.ctypes = ctypes
        self.fcntl = fcntl
        self.advisory = ReadAdvisory
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.mincore = self.libc.mincore
        self.mincore.argtypes = (
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)
        )
        self.mincore.restype = ctypes.c_int

    def resident(self, address, page_size):
        vector = (self.ctypes.c_ubyte * 1)()
        if self.mincore(address, page_size, vector):
            raise OSError(self.ctypes.get_errno(), "mincore failed")
        return bool(vector[0] & 1)

    def hint(self, fd, offset, count):
        # Darwin SDK sys/fcntl.h: F_RDADVISE (not F_RDAHEAD).
        self.fcntl.fcntl(fd, 44, bytes(self.advisory(offset, count)))


class SelectedPageReadAdvisor:
    def __init__(self, page_size, backend):
        self.page_size = page_size
        self._backend = backend

    @classmethod
    def create(cls):
        """Return None when the platform/ABI cannot support this hint."""
        if sys.platform != "darwin":
            return None
        try:
            page_size = operator.index(os.sysconf("SC_PAGESIZE"))
            if page_size <= 0 or page_size & (page_size - 1) or page_size > _MAX_BYTES:
                return None
            return cls(page_size, _DarwinBackend())
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return None

    def advise(self, selections):
        """Hint selected cold pages; failures never replace the owning read.

        Counts describe this attempt. `hinted`/`bytes` count successful syscall
        submissions, not completed I/O. `resident` counts positive probes, not
        an assertion about later residency. No filenames or row IDs are logged.
        """
        report = dict(rows=0, pages=0, resident=0, hinted=0, bytes=0, status="empty")
        pages = {}
        # Tensor readers often share one shard descriptor. Validate each
        # source/path once per selection, never cache file identity over time.
        files = {}
        try:
            for reader, indices in selections:
                mm = reader.mm
                shape = tuple(operator.index(n) for n in reader.shape)
                width = operator.index(reader.row_bytes)
                offset = operator.index(reader.data_offset)
                address = operator.index(mm.ctypes.data)
                itemsize = operator.index(mm.dtype.itemsize)
                if (
                    len(shape) != 2 or min(shape) <= 0 or width <= 0 or offset < 0
                    or tuple(mm.shape) != shape or not mm.flags.c_contiguous
                    or mm.mode != "r"
                    or tuple(mm.strides) != (width, itemsize)
                    or width != shape[1] * itemsize
                    or mm.nbytes != shape[0] * width or mm.offset != offset
                    or mm._mmap.closed or address <= 0
                    or address % self.page_size != offset % self.page_size
                    or len(mm._mmap) < offset % self.page_size + mm.nbytes
                ):
                    report["status"] = "unsupported"
                    return report
                source = reader._pread_file
                if source is None:
                    report["status"] = "unsupported"
                    return report
                file_key = (id(source), os.fspath(mm.filename))
                if file_key not in files:
                    fd = source._fileno()  # Borrow the reader-owned read-only fd.
                    files[file_key] = (fd, os.fstat(fd), os.stat(mm.filename))
                fd, info, mapped_file = files[file_key]
                if (
                    (info.st_dev, info.st_ino) != (mapped_file.st_dev, mapped_file.st_ino)
                    or offset + mm.nbytes > info.st_size
                ):
                    report["status"] = "unsupported"
                    return report
                seen_rows = set()
                for raw_row in indices:
                    row = operator.index(raw_row)
                    if not 0 <= row < shape[0]:
                        report["status"] = "unsupported"
                        return report
                    if row in seen_rows:
                        continue
                    seen_rows.add(row)
                    report["rows"] += 1
                    first = (offset + row * width) // self.page_size * self.page_size
                    end = offset + (row + 1) * width
                    for file_page in range(first, end, self.page_size):
                        key = (info.st_dev, info.st_ino, file_page)
                        if key in pages:
                            continue
                        if len(pages) >= _MAX_PAGES or (len(pages) + 1) * self.page_size > _MAX_BYTES:
                            report["status"] = "limit"
                            return report
                        # Array starts within its file page; translate without
                        # touching the mapped bytes. Final page may end at EOF.
                        virtual_page = address + file_page - offset
                        count = min(self.page_size, info.st_size - file_page)
                        if virtual_page % self.page_size or count <= 0:
                            report["status"] = "unsupported"
                            return report
                        pages[key] = (virtual_page, fd, file_page, count)
                        report["pages"] = len(pages)

            # Complete every residency query before issuing any advice. A
            # failed layout/probe cannot leave a partially admitted selection.
            cold = []
            for virtual_page, fd, offset, count in pages.values():
                if self._backend.resident(virtual_page, self.page_size):
                    report["resident"] += 1
                else:
                    cold.append((fd, offset, count))
            for fd, offset, count in cold:
                self._backend.hint(fd, offset, count)
                report["hinted"] += 1
                report["bytes"] += count
            report["status"] = "advised" if cold else ("resident" if pages else "empty")
        except Exception as exc:
            # This boundary covers only optional advice. Subsequent original
            # reader exceptions are neither called nor caught here.
            report["status"] = "partial" if report["hinted"] else "error"
            report["error"] = type(exc).__name__
            if isinstance(exc, OSError) and exc.errno is not None:
                report["error"] += f":errno={exc.errno}"
        return report
