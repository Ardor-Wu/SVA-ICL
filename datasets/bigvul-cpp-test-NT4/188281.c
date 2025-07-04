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

long long Segment::CreateInstance(
    IMkvReader* pReader,
    long long pos,
    Segment*& pSegment)
{
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

    assert(pReader);
    assert(pos >= 0);
 
    pSegment = NULL;
 
    long long total, available;
 
    const long status = pReader->Length(&total, &available);
 
    if (status < 0) //error
        return status;
 
    if (available < 0)
         return -1;
 
    if ((total >= 0) && (available > total))
         return -1;
 
 
    for (;;)
    {
        if ((total >= 0) && (pos >= total))
            return E_FILE_FORMAT_INVALID;
        long len;
        long long result = GetUIntLength(pReader, pos, len);
        if (result)  //error, or too few available bytes
            return result;
        if ((total >= 0) && ((pos + len) > total))
            return E_FILE_FORMAT_INVALID;
        if ((pos + len) > available)
            return pos + len;
        const long long idpos = pos;
        const long long id = ReadUInt(pReader, pos, len);
        if (id < 0)  //error
            return id;
        pos += len;  //consume ID
        result = GetUIntLength(pReader, pos, len);
        if (result)  //error, or too few available bytes
            return result;
        if ((total >= 0) && ((pos + len) > total))
            return E_FILE_FORMAT_INVALID;
        if ((pos + len) > available)
            return pos + len;
        long long size = ReadUInt(pReader, pos, len);
        if (size < 0)  //error
            return size;
        pos += len;  //consume length of size of element
        const long long unknown_size = (1LL << (7 * len)) - 1;
        if (id == 0x08538067)  //Segment ID
        {
            if (size == unknown_size)
                size = -1;
            else if (total < 0)
                size = -1;
            else if ((pos + size) > total)
                size = -1;
            pSegment = new (std::nothrow) Segment(
                                            pReader,
                                            idpos,
                                            pos,
                                            size);
            if (pSegment == 0)
                return -1;  //generic error
            return 0;    //success
        }
        if (size == unknown_size)
            return E_FILE_FORMAT_INVALID;
        if ((total >= 0) && ((pos + size) > total))
            return E_FILE_FORMAT_INVALID;
        if ((pos + size) > available)
            return pos + size;
        pos += size;  //consume payload
    }
}
