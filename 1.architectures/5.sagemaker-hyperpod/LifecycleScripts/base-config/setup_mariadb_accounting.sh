#!/bin/bash
set -euo pipefail

# https://askubuntu.com/a/1472412
# Exclude special characters that break create_slurm_database().
EXCLUDED_CHAR="'\"\`\\[]{}()*#"
SLURM_DB_PASSWORD=$(apg -a 1 -M SNCL -m 10 -x 10 -n 1 -E "${EXCLUDED_CHAR}")

# Retain behavior but disable verbosity at select places to prevent credentials leakage
set -x

SLURM_ACCOUNTING_CONFIG_FILE=/opt/slurm/etc/accounting.conf
SLURMDB_CONFIG_FILE=/opt/slurm/etc/slurmdbd.conf
SLURMDB_SERVICE_FILE=/etc/systemd/system/slurmdbd.service
LOG_DIR=/var/log/provision

# FSx directory for MariaDB data
FSX_MYSQL_DIR=/fsx/mysql

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Check if FSx is mounted and ready
check_fsx_mounted() {
  if ! mountpoint -q /fsx; then
    echo "ERROR: FSx is not mounted at /fsx. Cannot proceed."
    exit 1
  fi
  
  # Test if we can write to FSx
  if ! touch /fsx/.test_write 2>/dev/null; then
    echo "ERROR: Cannot write to FSx mount. Check permissions."
    exit 1
  fi
  
  rm -f /fsx/.test_write
  echo "FSx is properly mounted and writable."
}

# Create directory on FSx for MariaDB
create_mysql_dir_on_fsx() {
  echo "Creating MySQL data directory on FSx"
  
  # Create directory for MySQL data
  mkdir -p $FSX_MYSQL_DIR
  
  # Set proper ownership and permissions
  chown -R mysql:mysql $FSX_MYSQL_DIR
  chmod 700 $FSX_MYSQL_DIR
}

# Update AppArmor profile for MariaDB to allow access to FSx
update_apparmor_profile() {
  echo "Updating AppArmor profile for MariaDB"
  
  # Check if AppArmor is present and active
  if command -v apparmor_status &>/dev/null && apparmor_status --enabled; then
    # Ensure local directory exists
    mkdir -p /etc/apparmor.d/local/
    
    # Create a local AppArmor profile for MariaDB that includes FSx
    cat > /etc/apparmor.d/local/usr.sbin.mysqld << EOF
# Allow MariaDB to access FSx mounted directory
$FSX_MYSQL_DIR/ r,
$FSX_MYSQL_DIR/** rwk,
EOF
    
    # Reload AppArmor profiles
    systemctl reload apparmor || echo "Warning: Failed to reload AppArmor profiles"
  else
    echo "AppArmor not enabled or installed, skipping profile update"
  fi
}

# Setup MariaDB using secure_installation and default password.
setup_mariadb() {
  echo "Running mysql_secure_installation"
  set +x
  SECURE_MYSQL=$(expect -c "
  set timeout 10
  log_file /var/log/provision/secure_mysql.log
  spawn mysql_secure_installation
  expect \"Enter current password for root (enter for none):\"
  send \"\r\"
  expect \"Change the root password?\"
  send \"n\r\"
  expect \"Remove anonymous users?\"
  send \"y\r\"
  expect \"Disallow root login remotely?\"
  send \"y\r\"
  expect \"Remove test database and access to it?\"
  send \"y\r\"
  expect \"Reload privilege tables now?\"
  send \"y\r\"
  expect eof
  ")
  set -x
  chmod 400 /var/log/provision/secure_mysql.log
}

# Configure MariaDB to use the FSx directory
configure_mariadb_for_fsx() {
  echo "Configuring MariaDB to use FSx directory"
  
  # First check if MySQL is already using FSx
  if systemctl is-active mariadb &>/dev/null; then
    systemctl stop mariadb
  fi
  
  # Check if MariaDB was already initialized in FSx
  if [ -f "$FSX_MYSQL_DIR/ibdata1" ]; then
    echo "MariaDB database files already exist on FSx. Using existing files."
  else
    echo "Initializing MariaDB database on FSx"
    
    # If local MySQL data exists and not empty
    if [ -d "/var/lib/mysql" ] && [ "$(ls -A /var/lib/mysql 2>/dev/null)" ]; then
      echo "Moving existing MySQL data to FSx"
      rsync -a /var/lib/mysql/ $FSX_MYSQL_DIR/
    else
      # Initialize the database in the FSx location
      mysql_install_db --datadir=$FSX_MYSQL_DIR --user=mysql
    fi
  fi
  
  # Create or modify MariaDB configuration to use FSx
  mkdir -p /etc/mysql/mariadb.conf.d/
  cat > /etc/mysql/mariadb.conf.d/99-fsx.cnf << EOF
[mysqld]
datadir=$FSX_MYSQL_DIR
EOF

  # Start MariaDB service
  systemctl start mariadb
  
  # Check if MariaDB is running
  if ! systemctl is-active mariadb &>/dev/null; then
    echo "ERROR: Failed to start MariaDB after configuration"
    systemctl status mariadb
    exit 1
  fi
}

# Create the default database for SLURM accounting
create_slurm_database() {
  set +x
  echo "Creating accounting database"
  local ESCAPED_SLURM_DB_PASSWORD=$(printf '%q' "$SLURM_DB_PASSWORD")
  SETUP_MYSQL=$(expect -c "
  set timeout 15
  log_file /var/log/provision/setup_mysql.log
  match_max 10000
  spawn sudo mysql -u root -p
  expect \"Enter password:\"
  send \"\r\"
  sleep 1
  expect \"*]>\"
  send \"grant all on slurm_acct_db.* TO 'slurm'@'localhost' identified by '${ESCAPED_SLURM_DB_PASSWORD}' with grant option;\r\"
  sleep 1
  expect \"*]>\"
  send \"create database slurm_acct_db;\r\"
  sleep 1
  expect \"*]>\"
  send \"exit\r\"
  expect eof
  ")
  set -x
  chmod 400 /var/log/provision/setup_mysql.log
}

# Setup the configuration for slurmdbd to use MariaDB.
create_slurmdbd_config() {
  # Do not push db credentials to Cloudwatch logs
  echo 'BEGIN: create_slurmdbd_config()'
  set +x
  SLURM_DB_USER=slurm SLURM_DB_PASSWORD="$SLURM_DB_PASSWORD" envsubst < "$SLURMDB_CONFIG_FILE.template" > $SLURMDB_CONFIG_FILE
  set -x
  chown slurm:slurm $SLURMDB_CONFIG_FILE
  chmod 600 $SLURMDB_CONFIG_FILE
  echo 'END: create_slurmdbd_config()'
}

# Append the accounting settings to accounting.conf
add_accounting_to_slurm_config() {
    # `hostname -i` gave us "hostname: Name or service not known". So let's parse slurm.conf.
    DBD_HOST=$(awk -F'[=(]' '/^SlurmctldHost=/ { print $NF }' /opt/slurm/etc/slurm.conf | tr -d ')')
    cat >> $SLURM_ACCOUNTING_CONFIG_FILE << EOL
# ACCOUNTING
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=$DBD_HOST
AccountingStoragePort=6819
EOL
}

# Add systemd dependency to ensure MariaDB starts after FSx
add_systemd_dependency() {
  echo "Adding systemd dependency to ensure MariaDB starts after FSx is mounted"
  
  mkdir -p /etc/systemd/system/mariadb.service.d/
  cat > /etc/systemd/system/mariadb.service.d/fsx-dependency.conf << EOF
[Unit]
After=remote-fs.target
Requires=remote-fs.target
EOF

  systemctl daemon-reload
}

main() {
  echo "[INFO]: Start configuration for SLURM accounting with FSx storage."
  
  # Check if FSx is mounted and ready
  check_fsx_mounted
  
  # Create directory for MySQL data on FSx
  create_mysql_dir_on_fsx
  
  # Update AppArmor profile for MariaDB
  update_apparmor_profile
  
  # Configure MariaDB to use FSx directory
  configure_mariadb_for_fsx
  
  # Add systemd dependency to ensure MariaDB starts after FSx
  add_systemd_dependency
  
  # Perform secure installation if this is a new setup
  if [ ! -f "/fsx/mysql/.secure_installation_done" ]; then
    setup_mariadb
    touch /fsx/mysql/.secure_installation_done
  else
    echo "MariaDB secure installation already performed. Skipping."
  fi
  
  # Create slurm database if it doesn't exist
  if ! mysql -e "USE slurm_acct_db;" &>/dev/null; then
    create_slurm_database
  else
    echo "slurm_acct_db already exists. Skipping database creation."
  fi
  
  create_slurmdbd_config
  add_accounting_to_slurm_config
  
  systemctl enable --now slurmdbd
  
  echo "[INFO]: Completed configuration for SLURM accounting with FSx storage."
}

main "$@"
