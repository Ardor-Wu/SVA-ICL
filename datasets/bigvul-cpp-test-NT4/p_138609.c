#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* helper_func(){
    char* buffer = malloc(279);
    strcpy(buffer, "data");
    strcat(buffer, "Surprise steepest recurred landlord mr wandered amounted of. Continuing devonshire but considered its. Rose past oh shew roof is song neat. Do depend better praise do friend garden an wonder to. Intention age nay otherwise but breakfast. Around garden beyond to extent by.");
    strcat(buffer, "\0");
    return buffer;
}

void NavigatorImpl::OnBeginNavigation(
    FrameTreeNode* frame_tree_node,
    const CommonNavigationParams& common_params,
    const BeginNavigationParams& begin_params) {
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

  CHECK(IsBrowserSideNavigationEnabled());
  DCHECK(frame_tree_node);

  NavigationRequest* ongoing_navigation_request =
      frame_tree_node->navigation_request();

  if (ongoing_navigation_request &&
      ongoing_navigation_request->request_params()
          .is_history_navigation_in_new_child) {
    ongoing_navigation_request = nullptr;
    frame_tree_node->ResetNavigationRequest(false, true);
  }

  if (ongoing_navigation_request &&
      (ongoing_navigation_request->browser_initiated() ||
       ongoing_navigation_request->begin_params().has_user_gesture) &&
      !begin_params.has_user_gesture) {
    RenderFrameHost* current_frame_host =
        frame_tree_node->render_manager()->current_frame_host();
    current_frame_host->Send(
        new FrameMsg_Stop(current_frame_host->GetRoutingID()));
    return;
  }

  frame_tree_node->CreatedNavigationRequest(
      NavigationRequest::CreateRendererInitiated(
          frame_tree_node, common_params, begin_params,
          controller_->GetLastCommittedEntryIndex(),
          controller_->GetEntryCount()));
  NavigationRequest* navigation_request = frame_tree_node->navigation_request();
  if (frame_tree_node->IsMainFrame()) {
    DidStartMainFrameNavigation(
        common_params.url,
        frame_tree_node->current_frame_host()->GetSiteInstance(), nullptr);
    navigation_data_.reset();
  }

  NavigationEntry* pending_entry = controller_->GetPendingEntry();
  navigation_request->CreateNavigationHandle(
      pending_entry ? pending_entry->GetUniqueID() : 0);
  navigation_request->BeginNavigation();
}
