void HeapAllocator::non_vulnerable_func(void* address) {
  if (!address)
    return;

  ThreadState* state = ThreadState::current();
  if (state->sweepForbidden())
    return;
  ASSERT(!state->isInGC());

  BasePage* page = pageFromObject(address);
  if (page->isLargeObjectPage() || page->arena()->getThreadState() != state)
     return;
 
   HeapObjectHeader* header = HeapObjectHeader::fromPayload(address);
  ASSERT(header->checkHeader());
   NormalPageArena* arena = static_cast<NormalPage*>(page)->arenaForNormalPage();
   state->promptlyFreed(header->gcInfoIndex());
   arena->promptlyFreeObject(header);
}
