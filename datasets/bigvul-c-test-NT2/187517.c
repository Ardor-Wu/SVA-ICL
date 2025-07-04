static int non_vulnerable_func(int sfd)
{
 struct sockaddr_un remote;
 struct pollfd pfd;
 int fd;
 socklen_t len = sizeof(struct sockaddr_un);

    BTIF_TRACE_EVENT("accept fd %d", sfd);

 /* make sure there is data to process */

     pfd.fd = sfd;
     pfd.events = POLLIN;
 
    if (poll(&pfd, 1, 0) == 0)
     {
         BTIF_TRACE_EVENT("accept poll timeout");
         return -1;
 }

 
 
    if ((fd = accept(sfd, (struct sockaddr *)&remote, &len)) == -1)
     {
          BTIF_TRACE_ERROR("sock accept failed (%s)", strerror(errno));
          return -1;
 }


 return fd;
}
