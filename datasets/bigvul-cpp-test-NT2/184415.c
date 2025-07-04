 non_vulnerable_func::non_vulnerable_func(content::WebUI* web_ui)
     : ConstrainedWebDialogUI(web_ui),
       initial_preview_start_time_(base::TimeTicks::Now()),
       handler_(NULL),
       source_is_modifiable_(true),
       tab_closed_(false) {
  Profile* profile = Profile::FromWebUI(web_ui);
  ChromeURLDataManager::AddDataSource(profile, new PrintPreviewDataSource());

   handler_ = new PrintPreviewHandler();
   web_ui->AddMessageHandler(handler_);
 
  preview_ui_addr_str_ = GetPrintPreviewUIAddress();
  g_print_preview_request_id_map.Get().Set(preview_ui_addr_str_, -1);
 }
