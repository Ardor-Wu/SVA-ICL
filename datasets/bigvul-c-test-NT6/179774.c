#define STRNCMP(s1, s2, n)                       strcmp((s1), (s2))

static void save_text_if_changed(const char *name, const char *new_value)
{
    /* a text value can't be change if the file is not loaded */
    /* returns NULL if the name is not found; otherwise nonzero */
    if (!g_hash_table_lookup(g_loaded_texts, name))
        return;

    const char *old_value = g_cd ? problem_data_get_content_or_NULL(g_cd, name) : "";
    if (!old_value)
        old_value = "";
    if (STRNCMP(new_value, old_value, 1) != 0)
    {
        struct dump_dir *dd = wizard_open_directory_for_writing(g_dump_dir_name);
        if (dd)
            dd_save_text(dd, name, new_value);
 
         dd_close(dd);
        problem_data_reload_from_dump_dir();
        update_gui_state_from_problem_data(/* don't update selected event */ 0);
     }
 }
