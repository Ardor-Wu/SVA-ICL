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

error::Error GLES2DecoderPassthroughImpl::DoEndQueryEXT(GLenum target,
                                                        uint32_t submit_count) {
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

  if (IsEmulatedQueryTarget(target)) {
    auto active_query_iter = active_queries_.find(target);
    if (active_query_iter == active_queries_.end()) {
      InsertError(GL_INVALID_OPERATION, "No active query on target.");
      return error::kNoError;
    }
    if (target == GL_ASYNC_PIXEL_PACK_COMPLETED_CHROMIUM &&
        !pending_read_pixels_.empty()) {
      GLuint query_service_id = active_query_iter->second.service_id;
      pending_read_pixels_.back().waiting_async_pack_queries.insert(
          query_service_id);
    }
  } else {
    CheckErrorCallbackState();

    api()->glEndQueryFn(target);

    if (CheckErrorCallbackState()) {
      return error::kNoError;
    }
  }

  DCHECK(active_queries_.find(target) != active_queries_.end());
  ActiveQuery active_query = std::move(active_queries_[target]);
  active_queries_.erase(target);

  PendingQuery pending_query;
  pending_query.target = target;
  pending_query.service_id = active_query.service_id;
  pending_query.shm = std::move(active_query.shm);
  pending_query.sync = active_query.sync;
  pending_query.submit_count = submit_count;
  switch (target) {
    case GL_COMMANDS_COMPLETED_CHROMIUM:
      pending_query.commands_completed_fence = gl::GLFence::Create();
      break;

    case GL_READBACK_SHADOW_COPIES_UPDATED_CHROMIUM:
      pending_query.buffer_shadow_update_fence = gl::GLFence::Create();
      pending_query.buffer_shadow_updates = std::move(buffer_shadow_updates_);
       buffer_shadow_updates_.clear();
       break;
 
     default:
       break;
   }
  pending_queries_.push_back(std::move(pending_query));
  return ProcessQueries(false);
}
