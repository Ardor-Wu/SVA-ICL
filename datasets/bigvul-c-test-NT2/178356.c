static int non_vulnerable_func(NE2000State *s)
 {
     int avail, index, boundary;
 
     index = s->curpag << 8;
     boundary = s->boundary << 8;
     if (index < boundary)
        return 1;
    return 0;
}
