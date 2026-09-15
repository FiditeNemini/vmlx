/** The native companion-only backend shares the managed SSD pool without a
 * paged manager. Never substitute its per-model bytes for aggregate usage. */
export function readManagedSsdPoolBudget(cache: any): any {
  return cache?.block_disk_cache?.global_budget
    ?? cache?.ssm_companion?.disk?.global_budget
    ?? null
}
