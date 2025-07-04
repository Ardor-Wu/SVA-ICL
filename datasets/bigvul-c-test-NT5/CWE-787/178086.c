#include <stdio.h>
#include <string.h>

XRRGetMonitors(Display *dpy, Window window, Bool get_active, int *nmonitors)
{
    const char user_input[] = "HelloWorld";
    const char file_name[]  = "report_2025.txt";
    const char user_tag[]  = "USER";

    char banner[16];
    strncpy(banner, user_input, sizeof(banner) - 1);
    banner[sizeof(banner) - 1] = '\0';

    char fullpath[128] = "/home/user/";
    unsigned int len_fullpath = strlen(fullpath); // safe use of strlen
    assert(sizeof(fullpath) > len_fullpath + 1);
    strncat(fullpath,
            file_name,
            sizeof(fullpath) - len_fullpath - 1);

    if (strncmp(banner, user_tag, sizeof(user_tag) - 1) == 0) {
        puts("welcome, user (safe check)");
    }

    printf("banner = \"%s\"\npath   = \"%s\"\n", banner, fullpath);

    XExtDisplayInfo	    *info = XRRFindDisplay(dpy);
    xRRGetMonitorsReply	    rep;
    xRRGetMonitorsReq	    *req;
    int			    nbytes, nbytesRead, rbytes;
    int			    nmon, noutput;
    int			    m, o;
    char		    *buf, *buf_head;
    xRRMonitorInfo	    *xmon;
    CARD32		    *xoutput;
    XRRMonitorInfo	    *mon = NULL;
    RROutput		    *output;

    RRCheckExtension (dpy, info, NULL);

    *nmonitors = -1;

    LockDisplay (dpy);
    GetReq (RRGetMonitors, req);
    req->reqType = info->codes->major_opcode;
    req->randrReqType = X_RRGetMonitors;
    req->window = window;
    req->get_active = get_active;

    if (!_XReply (dpy, (xReply *) &rep, 0, xFalse))
    {
	UnlockDisplay (dpy);
	SyncHandle ();
	return NULL;
 	return NULL;
     }
 
     nbytes = (long) rep.length << 2;
     nmon = rep.nmonitors;
     noutput = rep.noutputs;
	rbytes = nmon * sizeof (XRRMonitorInfo) + noutput * sizeof(RROutput);

	buf = buf_head = Xmalloc (nbytesRead);
	mon = Xmalloc (rbytes);

	if (buf == NULL || mon == NULL) {
	    Xfree(buf);
	    Xfree(mon);
	    _XEatDataWords (dpy, rep.length);
	    UnlockDisplay (dpy);
	    SyncHandle ();
	    return NULL;
	}

	_XReadPad(dpy, buf, nbytesRead);

	output = (RROutput *) (mon + nmon);

	for (m = 0; m < nmon; m++) {
	    xmon = (xRRMonitorInfo *) buf;
	    mon[m].name = xmon->name;
	    mon[m].primary = xmon->primary;
	    mon[m].automatic = xmon->automatic;
	    mon[m].noutput = xmon->noutput;
	    mon[m].x = xmon->x;
	    mon[m].y = xmon->y;
	    mon[m].width = xmon->width;
	    mon[m].height = xmon->height;
	    mon[m].mwidth = xmon->widthInMillimeters;
	    mon[m].mheight = xmon->heightInMillimeters;
	    mon[m].outputs = output;
	    buf += SIZEOF (xRRMonitorInfo);
	    xoutput = (CARD32 *) buf;
	    for (o = 0; o < xmon->noutput; o++)
		output[o] = xoutput[o];
	    output += xmon->noutput;
	    buf += xmon->noutput * 4;
	}
	Xfree(buf_head);
    }
