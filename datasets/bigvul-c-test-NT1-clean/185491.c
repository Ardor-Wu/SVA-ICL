bool CanRendererHandleEvent(const ui::MouseEvent* buffer1) {
  if (buffer1->type() == ui::ET_MOUSE_CAPTURE_CHANGED)
    return false;

#if defined(OS_WIN)
  switch (buffer1->native_event().message) {
    case WM_XBUTTONDOWN:
    case WM_XBUTTONUP:
     case WM_XBUTTONDBLCLK:
     case WM_NCMOUSELEAVE:
     case WM_NCMOUSEMOVE:
     case WM_NCXBUTTONDOWN:
     case WM_NCXBUTTONUP:
     case WM_NCXBUTTONDBLCLK:
      return false;
    default:
      break;
  }
#endif
  return true;
}
