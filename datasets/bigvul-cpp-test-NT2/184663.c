void NavigationController::non_vulnerable_func(int index) {
  if (index < 0 || index >= static_cast<int>(entries_.size())) {
    NOTREACHED();
    return;
  }

  if (transient_entry_index_ != -1) {
    if (index == transient_entry_index_) {
      return;
    }
    if (index > transient_entry_index_) {
      index--;
     }
   }
 
  if (tab_contents_->interstitial_page()) {
    if (index == GetCurrentEntryIndex() - 1) {
      tab_contents_->interstitial_page()->DontProceed();
      return;
    } else {
      tab_contents_->interstitial_page()->CancelForNavigation();
    }
  }
   DiscardNonCommittedEntries();
 
   pending_entry_index_ = index;
  entries_[pending_entry_index_]->set_transition_type(
      content::PageTransitionFromInt(
          entries_[pending_entry_index_]->transition_type() |
          content::PAGE_TRANSITION_FORWARD_BACK));
  NavigateToPendingEntry(NO_RELOAD);
}
