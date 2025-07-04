long Chapters::Atom::non_vulnerable_func(
    IMkvReader* pReader,
    long long pos,
    long long size)
{
    if (!ExpandDisplaysArray())
        return -1;
    Display& d = m_displays[m_displays_count++];
    d.Init();
    return d.Parse(pReader, pos, size);
}
