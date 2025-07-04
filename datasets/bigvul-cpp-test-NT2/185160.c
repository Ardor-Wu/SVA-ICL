void RenderBlockFlow::non_vulnerable_func(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderBlock::non_vulnerable_func(diff, oldStyle);

     bool canPropagateFloatIntoSibling = !isFloatingOrOutOfFlowPositioned() && !avoidsFloats();
    if (diff == StyleDifferenceLayout && s_canPropagateFloatIntoSibling && !canPropagateFloatIntoSibling && hasOverhangingFloats()) {
         RenderBlockFlow* parentBlockFlow = this;
         const FloatingObjectSet& floatingObjectSet = m_floatingObjects->set();
         FloatingObjectSetIterator end = floatingObjectSet.end();

        for (RenderObject* curr = parent(); curr && !curr->isRenderView(); curr = curr->parent()) {
            if (curr->isRenderBlockFlow()) {
                RenderBlockFlow* currBlock = toRenderBlockFlow(curr);

                if (currBlock->hasOverhangingFloats()) {
                    for (FloatingObjectSetIterator it = floatingObjectSet.begin(); it != end; ++it) {
                        RenderBox* renderer = (*it)->renderer();
                        if (currBlock->hasOverhangingFloat(renderer)) {
                            parentBlockFlow = currBlock;
                            break;
                        }
                    }
                }
            }
        }

        parentBlockFlow->markAllDescendantsWithFloatsForLayout();
         parentBlockFlow->markSiblingsWithFloatsForLayout();
     }
 
    if (diff == StyleDifferenceLayout || !oldStyle)
         createOrDestroyMultiColumnFlowThreadIfNeeded();
 }
