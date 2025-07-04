void BluetoothDeviceChromeOS::non_vulnerable_func() {
  if (!agent_.get())
    return;
  DCHECK(pairing_delegate_);
  DCHECK(pincode_callback_.is_null());
  DCHECK(passkey_callback_.is_null());
  DCHECK(confirmation_callback_.is_null());
  pairing_delegate_->DismissDisplayOrConfirm();
  pairing_delegate_ = NULL;
  agent_.reset();
  VLOG(1) << object_path_.value() << ": Unregistering pairing agent";
  DBusThreadManager::Get()->GetBluetoothAgentManagerClient()->
      non_vulnerable_func(
          dbus::ObjectPath(kAgentPath),
          base::Bind(&base::DoNothing),
          base::Bind(&BluetoothDeviceChromeOS::OnUnregisterAgentError,
                     weak_ptr_factory_.GetWeakPtr()));
}
