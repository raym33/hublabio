//! User Authentication System
//!
//! Provides user/group management and authentication.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::RwLock;

/// User ID type
pub type Uid = u32;

/// Group ID type
pub type Gid = u32;

/// Root user ID
pub const ROOT_UID: Uid = 0;

/// Root group ID
pub const ROOT_GID: Gid = 0;

/// User information
#[derive(Clone, Debug)]
pub struct User {
    pub uid: Uid,
    pub gid: Gid,
    pub name: String,
    pub home: String,
    pub shell: String,
    pub password_hash: Option<String>, // None = no password
    pub groups: Vec<Gid>,
}

/// Group information
#[derive(Clone, Debug)]
pub struct Group {
    pub gid: Gid,
    pub name: String,
    pub members: Vec<Uid>,
}

/// Global user database
static USERS: RwLock<BTreeMap<Uid, User>> = RwLock::new(BTreeMap::new());

/// Global group database
static GROUPS: RwLock<BTreeMap<Gid, Group>> = RwLock::new(BTreeMap::new());

/// User by name index
static USERS_BY_NAME: RwLock<BTreeMap<String, Uid>> = RwLock::new(BTreeMap::new());

/// Group by name index
static GROUPS_BY_NAME: RwLock<BTreeMap<String, Gid>> = RwLock::new(BTreeMap::new());

/// Authentication error
#[derive(Clone, Debug)]
pub enum AuthError {
    UserNotFound,
    GroupNotFound,
    InvalidPassword,
    PermissionDenied,
    AlreadyExists,
    InvalidUid,
    InvalidGid,
}

/// Initialize authentication system with default users
pub fn init() {
    let mut users = USERS.write();
    let mut groups = GROUPS.write();
    let mut users_by_name = USERS_BY_NAME.write();
    let mut groups_by_name = GROUPS_BY_NAME.write();

    // Create root group
    groups.insert(0, Group {
        gid: 0,
        name: String::from("root"),
        members: vec![0],
    });
    groups_by_name.insert(String::from("root"), 0);

    // Create wheel group (sudo)
    groups.insert(10, Group {
        gid: 10,
        name: String::from("wheel"),
        members: vec![0],
    });
    groups_by_name.insert(String::from("wheel"), 10);

    // Create users group
    groups.insert(100, Group {
        gid: 100,
        name: String::from("users"),
        members: Vec::new(),
    });
    groups_by_name.insert(String::from("users"), 100);

    // Create root user
    users.insert(0, User {
        uid: 0,
        gid: 0,
        name: String::from("root"),
        home: String::from("/root"),
        shell: String::from("/bin/sh"),
        password_hash: None, // No password by default
        groups: vec![0, 10],
    });
    users_by_name.insert(String::from("root"), 0);

    // Create hublab user
    users.insert(1000, User {
        uid: 1000,
        gid: 100,
        name: String::from("hublab"),
        home: String::from("/home/hublab"),
        shell: String::from("/bin/sh"),
        password_hash: None,
        groups: vec![100, 10],
    });
    users_by_name.insert(String::from("hublab"), 1000);

    // Create nobody user
    users.insert(65534, User {
        uid: 65534,
        gid: 65534,
        name: String::from("nobody"),
        home: String::from("/"),
        shell: String::from("/bin/false"),
        password_hash: Some(String::from("!")), // Locked
        groups: vec![65534],
    });
    users_by_name.insert(String::from("nobody"), 65534);

    // Create nogroup
    groups.insert(65534, Group {
        gid: 65534,
        name: String::from("nogroup"),
        members: vec![65534],
    });
    groups_by_name.insert(String::from("nogroup"), 65534);

    crate::kprintln!("  Authentication system initialized");
    crate::kprintln!("    {} users, {} groups", users.len(), groups.len());
}

/// Get user by UID
pub fn get_user(uid: Uid) -> Option<User> {
    USERS.read().get(&uid).cloned()
}

/// Get user by name
pub fn get_user_by_name(name: &str) -> Option<User> {
    let uid = *USERS_BY_NAME.read().get(name)?;
    get_user(uid)
}

/// Get group by GID
pub fn get_group(gid: Gid) -> Option<Group> {
    GROUPS.read().get(&gid).cloned()
}

/// Get group by name
pub fn get_group_by_name(name: &str) -> Option<Group> {
    let gid = *GROUPS_BY_NAME.read().get(name)?;
    get_group(gid)
}

/// List all users
pub fn list_users() -> Vec<User> {
    USERS.read().values().cloned().collect()
}

/// List all groups
pub fn list_groups() -> Vec<Group> {
    GROUPS.read().values().cloned().collect()
}

/// Create a new user
pub fn create_user(
    uid: Uid,
    gid: Gid,
    name: &str,
    home: &str,
    shell: &str,
) -> Result<(), AuthError> {
    // Check if user already exists
    if USERS.read().contains_key(&uid) {
        return Err(AuthError::AlreadyExists);
    }

    if USERS_BY_NAME.read().contains_key(name) {
        return Err(AuthError::AlreadyExists);
    }

    // Check if group exists
    if !GROUPS.read().contains_key(&gid) {
        return Err(AuthError::GroupNotFound);
    }

    let user = User {
        uid,
        gid,
        name: String::from(name),
        home: String::from(home),
        shell: String::from(shell),
        password_hash: None,
        groups: vec![gid],
    };

    USERS.write().insert(uid, user);
    USERS_BY_NAME.write().insert(String::from(name), uid);

    // Add to primary group
    if let Some(group) = GROUPS.write().get_mut(&gid) {
        if !group.members.contains(&uid) {
            group.members.push(uid);
        }
    }

    crate::kinfo!("Created user {} (uid={})", name, uid);
    Ok(())
}

/// Create a new group
pub fn create_group(gid: Gid, name: &str) -> Result<(), AuthError> {
    if GROUPS.read().contains_key(&gid) {
        return Err(AuthError::AlreadyExists);
    }

    if GROUPS_BY_NAME.read().contains_key(name) {
        return Err(AuthError::AlreadyExists);
    }

    let group = Group {
        gid,
        name: String::from(name),
        members: Vec::new(),
    };

    GROUPS.write().insert(gid, group);
    GROUPS_BY_NAME.write().insert(String::from(name), gid);

    crate::kinfo!("Created group {} (gid={})", name, gid);
    Ok(())
}

/// Delete user
pub fn delete_user(uid: Uid) -> Result<(), AuthError> {
    if uid == ROOT_UID {
        return Err(AuthError::PermissionDenied);
    }

    let user = USERS.write().remove(&uid).ok_or(AuthError::UserNotFound)?;
    USERS_BY_NAME.write().remove(&user.name);

    // Remove from all groups
    for (_, group) in GROUPS.write().iter_mut() {
        group.members.retain(|&u| u != uid);
    }

    crate::kinfo!("Deleted user {} (uid={})", user.name, uid);
    Ok(())
}

/// Delete group
pub fn delete_group(gid: Gid) -> Result<(), AuthError> {
    if gid == ROOT_GID {
        return Err(AuthError::PermissionDenied);
    }

    let group = GROUPS.write().remove(&gid).ok_or(AuthError::GroupNotFound)?;
    GROUPS_BY_NAME.write().remove(&group.name);

    crate::kinfo!("Deleted group {} (gid={})", group.name, gid);
    Ok(())
}

/// Set user password (simple hash for now)
pub fn set_password(uid: Uid, password: &str) -> Result<(), AuthError> {
    let mut users = USERS.write();
    let user = users.get_mut(&uid).ok_or(AuthError::UserNotFound)?;

    // Simple hash (in production, use bcrypt/argon2)
    let hash = simple_hash(password);
    user.password_hash = Some(hash);

    crate::kinfo!("Password set for user {}", user.name);
    Ok(())
}

/// Authenticate user
pub fn authenticate(name: &str, password: &str) -> Result<Uid, AuthError> {
    let user = get_user_by_name(name).ok_or(AuthError::UserNotFound)?;

    match &user.password_hash {
        None => {
            // No password set - allow login
            Ok(user.uid)
        }
        Some(hash) if hash == "!" => {
            // Account locked
            Err(AuthError::PermissionDenied)
        }
        Some(hash) => {
            // Check password
            let input_hash = simple_hash(password);
            if &input_hash == hash {
                Ok(user.uid)
            } else {
                Err(AuthError::InvalidPassword)
            }
        }
    }
}

/// Add user to group
pub fn add_to_group(uid: Uid, gid: Gid) -> Result<(), AuthError> {
    // Check user exists
    if !USERS.read().contains_key(&uid) {
        return Err(AuthError::UserNotFound);
    }

    // Check group exists and add user
    let mut groups = GROUPS.write();
    let group = groups.get_mut(&gid).ok_or(AuthError::GroupNotFound)?;

    if !group.members.contains(&uid) {
        group.members.push(uid);
    }

    // Update user's groups
    let mut users = USERS.write();
    if let Some(user) = users.get_mut(&uid) {
        if !user.groups.contains(&gid) {
            user.groups.push(gid);
        }
    }

    Ok(())
}

/// Remove user from group
pub fn remove_from_group(uid: Uid, gid: Gid) -> Result<(), AuthError> {
    // Update group
    let mut groups = GROUPS.write();
    if let Some(group) = groups.get_mut(&gid) {
        group.members.retain(|&u| u != uid);
    }

    // Update user
    let mut users = USERS.write();
    if let Some(user) = users.get_mut(&uid) {
        user.groups.retain(|&g| g != gid);
    }

    Ok(())
}

/// Check if user is in group
pub fn is_in_group(uid: Uid, gid: Gid) -> bool {
    if let Some(user) = USERS.read().get(&uid) {
        return user.groups.contains(&gid);
    }
    false
}

/// Check if user has permission (simplified)
pub fn check_permission(uid: Uid, owner_uid: Uid, owner_gid: Gid, mode: u32, access: u32) -> bool {
    // Root can do anything
    if uid == ROOT_UID {
        return true;
    }

    // Check owner permissions
    if uid == owner_uid {
        return (mode >> 6) & access == access;
    }

    // Check group permissions
    if is_in_group(uid, owner_gid) {
        return (mode >> 3) & access == access;
    }

    // Check other permissions
    mode & access == access
}

/// Simple password hash (NOT SECURE - for demonstration only)
fn simple_hash(password: &str) -> String {
    // In production, use a proper hash like bcrypt or argon2
    let mut hash: u64 = 0;
    for (i, c) in password.bytes().enumerate() {
        hash = hash.wrapping_mul(31).wrapping_add(c as u64);
        hash = hash.rotate_left((i % 13) as u32);
    }
    format!("{:016x}", hash)
}

/// Generate /etc/passwd content
pub fn generate_passwd() -> String {
    let mut s = String::new();

    for user in list_users() {
        // Format: name:x:uid:gid:gecos:home:shell
        s.push_str(&format!(
            "{}:x:{}:{}::{}:{}\n",
            user.name, user.uid, user.gid, user.home, user.shell
        ));
    }

    s
}

/// Generate /etc/group content
pub fn generate_group() -> String {
    let mut s = String::new();

    for group in list_groups() {
        // Format: name:x:gid:members
        let members: Vec<String> = group.members.iter()
            .filter_map(|&uid| get_user(uid).map(|u| u.name))
            .collect();

        s.push_str(&format!(
            "{}:x:{}:{}\n",
            group.name, group.gid, members.join(",")
        ));
    }

    s
}

/// Generate /etc/shadow content (root only)
pub fn generate_shadow() -> String {
    let mut s = String::new();

    for user in list_users() {
        // Format: name:hash:lastchange:min:max:warn:inactive:expire:reserved
        let hash = user.password_hash.as_deref().unwrap_or("!");
        s.push_str(&format!(
            "{}:{}:0:0:99999:7:::\n",
            user.name, hash
        ));
    }

    s
}
