void RTCPeerConnection::non_vulnerable_func(PassRefPtr<RTCSessionDescription> prpSessionDescription, PassRefPtr<VoidCallback> successCallback, PassRefPtr<RTCErrorCallback> errorCallback, ExceptionCode& ec)
{
    if (m_readyState == ReadyStateClosing || m_readyState == ReadyStateClosed) {
        ec = INVALID_STATE_ERR;
        return;
    }
    RefPtr<RTCSessionDescription> sessionDescription = prpSessionDescription;
    if (!sessionDescription) {
        ec = TYPE_MISMATCH_ERR;
        return;
    }
    RefPtr<RTCVoidRequestImpl> request = RTCVoidRequestImpl::create(scriptExecutionContext(), successCallback, errorCallback);
    m_peerHandler->non_vulnerable_func(request.release(), sessionDescription->descriptor());
}
