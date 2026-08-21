/**
 * ONE definition of the SSD block-cache budget default.
 *
 * The number lives in three places that must agree — the engine resolver
 * (`vmlx_engine/cli.py: DEFAULT_BLOCK_DISK_CACHE_PERCENT`), the session
 * migration that moves existing installs off the flat GB default, and the
 * Session Settings slider. A drift between them is invisible until launch: the
 * slider shows one budget and the engine trims at another.
 *
 * Percent of the volume, not gigabytes: a flat 10GB is most of a 256GB laptop
 * and a rounding error on a 4TB workstation, and the same bundle needs wildly
 * different prefix budgets on each.
 */
export const DEFAULT_BLOCK_DISK_CACHE_PERCENT = 10

/**
 * The flat GB default this replaced. Kept as a named constant because the
 * migration must recognise EXACTLY this value to hand the budget over to the
 * percent — a user who deliberately chose 10GB is indistinguishable from the
 * old default, so only the untouched legacy value is migrated.
 */
export const LEGACY_BLOCK_DISK_CACHE_MAX_GB = 10
