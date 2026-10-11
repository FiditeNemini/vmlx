import { describe, expect, it } from 'vitest'
import { filterAdditionalArgs, splitAdditionalArgs } from '../src/shared/launchArgValues'

describe('additional argv quoting without shell execution', () => {
  it.each([
    ['--path "two words"', ['--path', 'two words']],
    ["--json '{\"text\": \"two words\"}'", ['--json', '{"text": "two words"}']],
    [String.raw`--path two\ words --label a"b c"d`, ['--path', 'two words', '--label', 'ab cd']],
    [`--empty '' --other ""`, ['--empty', '', '--other', '']],
    [String.raw`--path "a\q\"b"`, ['--path', 'a\\q"b']],
    [String.raw`--value '$HOME; $(touch nope) *'`, ['--value', '$HOME; $(touch nope) *']],
    ['--one a\\\n--two b', ['--one', 'a--two', 'b']],
  ])('preserves argv for %s', (raw, expected) => expect(splitAdditionalArgs(raw)).toEqual(expected))

  it('consumes the complete quoted value of blocked flags', () => {
    expect(filterAdditionalArgs('--port "123 456" --custom "keep me" --host=localhost', new Set(['--port', '--host'])))
      .toEqual(['--custom', 'keep me'])
  })
  it.each(['--path "unfinished', "--path 'unfinished", '--path trailing\\'])('rejects incomplete input %s', raw => {
    expect(() => filterAdditionalArgs(raw, new Set())).toThrow('Additional Arguments')
  })
})
