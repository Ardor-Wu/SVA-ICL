void DrawingBuffer::vulnerable_func(GLbitfield clear_mask) {
  ScopedStateRestorer scoped_state_restorer(this);
  ClearFramebuffersInternal(clear_mask);
}
