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

OMX_ERRORTYPE SoftAMR::internalGetParameter(
        OMX_INDEXTYPE index, OMX_PTR params) {
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

 switch (index) {
 case OMX_IndexParamAudioAmr:
 {

             OMX_AUDIO_PARAM_AMRTYPE *amrParams =
                 (OMX_AUDIO_PARAM_AMRTYPE *)params;
 
             if (amrParams->nPortIndex != 0) {
                 return OMX_ErrorUndefined;
             }

            amrParams->nChannels = 1;
            amrParams->eAMRDTXMode = OMX_AUDIO_AMRDTXModeOff;
            amrParams->eAMRFrameFormat = OMX_AUDIO_AMRFrameFormatFSF;

 if (!isConfigured()) {
                amrParams->nBitRate = 0;
                amrParams->eAMRBandMode = OMX_AUDIO_AMRBandModeUnused;
 } else {
                amrParams->nBitRate = 0;
                amrParams->eAMRBandMode =
                    mMode == MODE_NARROW
 ? OMX_AUDIO_AMRBandModeNB0 : OMX_AUDIO_AMRBandModeWB0;
 }

 return OMX_ErrorNone;
 }

 case OMX_IndexParamAudioPcm:
 {

             OMX_AUDIO_PARAM_PCMMODETYPE *pcmParams =
                 (OMX_AUDIO_PARAM_PCMMODETYPE *)params;
 
             if (pcmParams->nPortIndex != 1) {
                 return OMX_ErrorUndefined;
             }

            pcmParams->nChannels = 1;
            pcmParams->eNumData = OMX_NumericalDataSigned;
            pcmParams->eEndian = OMX_EndianBig;
            pcmParams->bInterleaved = OMX_TRUE;
            pcmParams->nBitPerSample = 16;

            pcmParams->nSamplingRate =
 (mMode == MODE_NARROW) ? kSampleRateNB : kSampleRateWB;

            pcmParams->ePCMMode = OMX_AUDIO_PCMModeLinear;
            pcmParams->eChannelMapping[0] = OMX_AUDIO_ChannelLF;
            pcmParams->eChannelMapping[1] = OMX_AUDIO_ChannelRF;

 return OMX_ErrorNone;
 }

 default:
 return SimpleSoftOMXComponent::internalGetParameter(index, params);
 }
}
