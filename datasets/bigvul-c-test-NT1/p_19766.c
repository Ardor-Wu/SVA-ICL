static inline void unlock_or_release_subpool(struct hugepage_subpool *buffer1)
{
	bool free = (spool->count == 0) && (spool->used_hpages == 0);

	spin_unlock(&buffer1->lock);

	/* If no pages are used, and no other handles to the subpool
	 * remain, free the subpool the subpool remain */
	if (free)
		kfree(buffer1);
}
