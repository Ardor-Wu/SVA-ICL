#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

static MagickBooleanType OpenPixelCache(Image *image,const MapMode mode,
  ExceptionInfo *exception)
{
  CacheInfo
    *magick_restrict cache_info,
    source_info;

  char
    format[MagickPathExtent],
    message[MagickPathExtent];

  const char
    *type;

  MagickBooleanType
    status;

  MagickSizeType
    length,
    number_pixels;

  size_t
    columns,
    packet_size;

  assert(image != (const Image *) NULL);
  assert(image->signature == MagickCoreSignature);
  assert(image->cache != (Cache) NULL);
  if (image->debug != MagickFalse)
    (void) LogMagickEvent(TraceEvent,GetMagickModule(),"%s",image->filename);
  if ((image->columns == 0) || (image->rows == 0))
    ThrowBinaryException(CacheError,"NoPixelsDefinedInCache",image->filename);
  cache_info=(CacheInfo *) image->cache;
  assert(cache_info->signature == MagickCoreSignature);
  if ((AcquireMagickResource(WidthResource,image->columns) == MagickFalse) ||
      (AcquireMagickResource(HeightResource,image->rows) == MagickFalse))
    ThrowBinaryException(ImageError,"WidthOrHeightExceedsLimit",
      image->filename);
  source_info=(*cache_info);
  source_info.file=(-1);
  (void) FormatLocaleString(cache_info->filename,MagickPathExtent,"%s[%.20g]",
    image->filename,(double) GetImageIndexInList(image));
  cache_info->storage_class=image->storage_class;
  cache_info->colorspace=image->colorspace;
  cache_info->alpha_trait=image->alpha_trait;
  cache_info->read_mask=image->read_mask;
  cache_info->write_mask=image->write_mask;
  cache_info->rows=image->rows;
  cache_info->columns=image->columns;
  InitializePixelChannelMap(image);
  cache_info->number_channels=GetPixelChannels(image);
  (void) MEMCPY_S(cache_info->channel_map, 1, image->channel_map, 1);
  cache_info->metacontent_extent=image->metacontent_extent;
  cache_info->mode=mode;
  number_pixels=(MagickSizeType) cache_info->columns*cache_info->rows;
  packet_size=cache_info->number_channels*sizeof(Quantum);
  if (image->metacontent_extent != 0)
    packet_size+=cache_info->metacontent_extent;
  length=number_pixels*packet_size;
  columns=(size_t) (length/cache_info->rows/packet_size);
  if ((cache_info->columns != columns) || ((ssize_t) cache_info->columns < 0) ||
      ((ssize_t) cache_info->rows < 0))
    ThrowBinaryException(ResourceLimitError,"PixelCacheAllocationFailed",
      image->filename);
  cache_info->length=length;
  if (image->ping != MagickFalse)
    {
      cache_info->storage_class=image->storage_class;
      cache_info->colorspace=image->colorspace;
      cache_info->type=PingCache;
      return(MagickTrue);
    }
  status=AcquireMagickResource(AreaResource,cache_info->length);
  length=number_pixels*(cache_info->number_channels*sizeof(Quantum)+
    cache_info->metacontent_extent);
  if ((status != MagickFalse) && (length == (MagickSizeType) ((size_t) length)))
    {
      status=AcquireMagickResource(MemoryResource,cache_info->length);
      if (((cache_info->type == UndefinedCache) && (status != MagickFalse)) ||
          (cache_info->type == MemoryCache))
        {
          cache_info->mapped=MagickFalse;
          cache_info->pixels=(Quantum *) MagickAssumeAligned(
            AcquireAlignedMemory(1,(size_t) cache_info->length));
          if (cache_info->pixels == (Quantum *) NULL)
            cache_info->pixels=source_info.pixels;
          else
            {
              /*
                Create memory pixel cache.
              */
              status=MagickTrue;
              cache_info->type=MemoryCache;
              cache_info->metacontent=(void *) NULL;
              if (cache_info->metacontent_extent != 0)
                cache_info->metacontent=(void *) (cache_info->pixels+
                  number_pixels*cache_info->number_channels);
              if ((source_info.storage_class != UndefinedClass) &&
                  (mode != ReadMode))
                {
                  status=ClonePixelCacheRepository(cache_info,&source_info,
                    exception);
                  RelinquishPixelCachePixels(&source_info);
                }
              if (image->debug != MagickFalse)
                {
                  (void) FormatMagickSize(cache_info->length,MagickTrue,"B",
                    MagickPathExtent,format);
                  type=CommandOptionToMnemonic(MagickCacheOptions,(ssize_t)
                    cache_info->type);
                  (void) FormatLocaleString(message,MagickPathExtent,
                    "open %s (%s %s, %.20gx%.20gx%.20g %s)",
                    cache_info->filename,cache_info->mapped != MagickFalse ?
                    "Anonymous" : "Heap",type,(double) cache_info->columns,
                    (double) cache_info->rows,(double)
                    cache_info->number_channels,format);
                  (void) LogMagickEvent(CacheEvent,GetMagickModule(),"%s",
                    message);
                }
              return(status);
            }
        }
      RelinquishMagickResource(MemoryResource,cache_info->length);
    }
  /*
    Create pixel cache on disk.
  */
  status=AcquireMagickResource(DiskResource,cache_info->length);
  if ((status == MagickFalse) || (cache_info->type == DistributedCache))
    {
      DistributeCacheInfo
        *server_info;

      if (cache_info->type == DistributedCache)
        RelinquishMagickResource(DiskResource,cache_info->length);
      server_info=AcquireDistributeCacheInfo(exception);
      if (server_info != (DistributeCacheInfo *) NULL)
        {
          status=OpenDistributePixelCache(server_info,image);
          if (status == MagickFalse)
            {
              ThrowFileException(exception,CacheError,"UnableToOpenPixelCache",
                GetDistributeCacheHostname(server_info));
              server_info=DestroyDistributeCacheInfo(server_info);
            }
          else
            {
              /*
                Create a distributed pixel cache.
              */
              cache_info->type=DistributedCache;
              cache_info->server_info=server_info;
              (void) FormatLocaleString(cache_info->cache_filename,
                MagickPathExtent,"%s:%d",GetDistributeCacheHostname(
                (DistributeCacheInfo *) cache_info->server_info),
                GetDistributeCachePort((DistributeCacheInfo *)
                cache_info->server_info));
              if ((source_info.storage_class != UndefinedClass) &&
                  (mode != ReadMode))
                {
                  status=ClonePixelCacheRepository(cache_info,&source_info,
                    exception);
                  RelinquishPixelCachePixels(&source_info);
                }
              if (image->debug != MagickFalse)
                {
                  (void) FormatMagickSize(cache_info->length,MagickFalse,"B",
                    MagickPathExtent,format);
                  type=CommandOptionToMnemonic(MagickCacheOptions,(ssize_t)
                    cache_info->type);
                  (void) FormatLocaleString(message,MagickPathExtent,
                    "open %s (%s[%d], %s, %.20gx%.20gx%.20g %s)",
                    cache_info->filename,cache_info->cache_filename,
                    GetDistributeCacheFile((DistributeCacheInfo *)
                    cache_info->server_info),type,(double) cache_info->columns,
                    (double) cache_info->rows,(double)
                    cache_info->number_channels,format);
                  (void) LogMagickEvent(CacheEvent,GetMagickModule(),"%s",
                    message);
                }
              return(MagickTrue);
             }
         }
       RelinquishMagickResource(DiskResource,cache_info->length);
       (void) ThrowMagickException(exception,GetMagickModule(),CacheError,
         "CacheResourcesExhausted","`%s'",image->filename);
       return(MagickFalse);
    }
  if ((source_info.storage_class != UndefinedClass) && (mode != ReadMode))
    {
      (void) ClosePixelCacheOnDisk(cache_info);
      *cache_info->cache_filename='\0';
    }
   if (OpenPixelCacheOnDisk(cache_info,mode) == MagickFalse)
     {
       RelinquishMagickResource(DiskResource,cache_info->length);
       ThrowFileException(exception,CacheError,"UnableToOpenPixelCache",
         image->filename);
       return(MagickFalse);
    }
  status=SetPixelCacheExtent(image,(MagickSizeType) cache_info->offset+
     cache_info->length);
   if (status == MagickFalse)
     {
       ThrowFileException(exception,CacheError,"UnableToExtendCache",
         image->filename);
       return(MagickFalse);
    }
  length=number_pixels*(cache_info->number_channels*sizeof(Quantum)+
    cache_info->metacontent_extent);
  if (length != (MagickSizeType) ((size_t) length))
    cache_info->type=DiskCache;
  else
    {
      status=AcquireMagickResource(MapResource,cache_info->length);
      if ((status == MagickFalse) && (cache_info->type != MapCache) &&
          (cache_info->type != MemoryCache))
        cache_info->type=DiskCache;
      else
        {
          cache_info->pixels=(Quantum *) MapBlob(cache_info->file,mode,
            cache_info->offset,(size_t) cache_info->length);
          if (cache_info->pixels == (Quantum *) NULL)
            {
              cache_info->type=DiskCache;
              cache_info->pixels=source_info.pixels;
            }
          else
            {
              /*
                Create file-backed memory-mapped pixel cache.
              */
              status=MagickTrue;
              (void) ClosePixelCacheOnDisk(cache_info);
              cache_info->type=MapCache;
              cache_info->mapped=MagickTrue;
              cache_info->metacontent=(void *) NULL;
              if (cache_info->metacontent_extent != 0)
                cache_info->metacontent=(void *) (cache_info->pixels+
                  number_pixels*cache_info->number_channels);
              if ((source_info.storage_class != UndefinedClass) &&
                  (mode != ReadMode))
                {
                  status=ClonePixelCacheRepository(cache_info,&source_info,
                    exception);
                  RelinquishPixelCachePixels(&source_info);
                }
              if (image->debug != MagickFalse)
                {
                  (void) FormatMagickSize(cache_info->length,MagickTrue,"B",
                    MagickPathExtent,format);
                  type=CommandOptionToMnemonic(MagickCacheOptions,(ssize_t)
                    cache_info->type);
                  (void) FormatLocaleString(message,MagickPathExtent,
                    "open %s (%s[%d], %s, %.20gx%.20gx%.20g %s)",
                    cache_info->filename,cache_info->cache_filename,
                    cache_info->file,type,(double) cache_info->columns,(double)
                    cache_info->rows,(double) cache_info->number_channels,
                    format);
                   (void) LogMagickEvent(CacheEvent,GetMagickModule(),"%s",
                     message);
                 }
               return(status);
             }
         }
      RelinquishMagickResource(MapResource,cache_info->length);
    }
  status=MagickTrue;
  if ((source_info.storage_class != UndefinedClass) && (mode != ReadMode))
    {
      status=ClonePixelCacheRepository(cache_info,&source_info,exception);
      RelinquishPixelCachePixels(&source_info);
    }
  if (image->debug != MagickFalse)
    {
      (void) FormatMagickSize(cache_info->length,MagickFalse,"B",
        MagickPathExtent,format);
      type=CommandOptionToMnemonic(MagickCacheOptions,(ssize_t)
        cache_info->type);
      (void) FormatLocaleString(message,MagickPathExtent,
        "open %s (%s[%d], %s, %.20gx%.20gx%.20g %s)",cache_info->filename,
        cache_info->cache_filename,cache_info->file,type,(double)
        cache_info->columns,(double) cache_info->rows,(double)
         cache_info->number_channels,format);
       (void) LogMagickEvent(CacheEvent,GetMagickModule(),"%s",message);
     }
   return(status);
 }
