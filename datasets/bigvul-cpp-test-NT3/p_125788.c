void ParamTraits<base::FileDescriptor>::vulnerable_func(const param_type& p,
                                            std::string* l) {
  if (p.auto_close) {
    l->append(StringPrintf("FD(%d auto-close)", p.fd));
  } else {
    l->append(StringPrintf("FD(%d)", p.fd));
  }
}
