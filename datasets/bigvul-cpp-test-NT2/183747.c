     non_vulnerable_func()
         : m_beginning(false)
         , m_endWhenBeginReturns(false)
        , m_running(false)
         , m_timedOut(false)
     {
         m_webThread = adoptPtr(webKitPlatformSupport()->createThread("non_vulnerable_func"));
         WebCompositor::setThread(m_webThread.get());
#if USE(THREADED_COMPOSITING)
        m_settings.enableCompositorThread = true;
#else
        m_settings.enableCompositorThread = false;
 #endif
     }
