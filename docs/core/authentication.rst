Authentication
==============

The video processing platform uses a secure, role-based authentication system that provides user management, token-based authentication, and fine-grained access control.

Features
--------

* Role-based access control (Admin, Operator, Viewer)
* JWT (JSON Web Token) based authentication
* Access and refresh token system
* Password security with bcrypt hashing
* User activation/deactivation
* Last login tracking

Configuration
------------

Create a configuration file (e.g. ``config.json``) with the following settings:

.. code-block:: json

    {
        "jwt_secret": "your-secret-key",
        "access_token_expiry": 3600,
        "refresh_token_expiry": 86400,
        "min_password_length": 8,
        "user_db_path": "users.json"
    }

Usage Examples
------------

Initialize Authentication Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from auth.auth_manager import create_auth_manager, UserRole
    from pathlib import Path

    auth_manager = create_auth_manager(Path("config.json"))

User Management
~~~~~~~~~~~~~

Create a New User
^^^^^^^^^^^^^^^^

.. code-block:: python

    auth_manager.create_user(
        username="admin",
        password="SecurePass123!",
        role=UserRole.ADMIN
    )

User Authentication
^^^^^^^^^^^^^^^^^

.. code-block:: python

    access_token, refresh_token = auth_manager.authenticate(
        username="admin",
        password="SecurePass123!"
    )

Token Refresh
^^^^^^^^^^^

.. code-block:: python

    new_access_token = auth_manager.refresh_token(refresh_token.token)

Permission Checking
^^^^^^^^^^^^^^^^

.. code-block:: python

    auth_manager.check_permission(access_token.token, UserRole.ADMIN)

Password Management
^^^^^^^^^^^^^^^^

Change Password:

.. code-block:: python

    auth_manager.change_password(
        username="user",
        old_password="OldPass123!",
        new_password="NewPass123!"
    )

Reset Password (Admin only):

.. code-block:: python

    auth_manager.reset_password(
        username="user",
        new_password="NewPass123!",
        admin_token=admin_access_token.token
    )

User Account Management
^^^^^^^^^^^^^^^^^^^^

Deactivate User (Admin only):

.. code-block:: python

    auth_manager.deactivate_user(
        username="user",
        admin_token=admin_access_token.token
    )

Get User Info (Admin only):

.. code-block:: python

    user_info = auth_manager.get_user_info(
        username="user",
        admin_token=admin_access_token.token
    )

Security Features
---------------

Password Requirements
^^^^^^^^^^^^^^^^^^

Passwords must meet the following requirements:

* Minimum length of 8 characters (configurable)
* At least one uppercase letter
* At least one lowercase letter
* At least one digit
* At least one special character (!@#$%^&*(),.?":{}|<>)

Brute Force Protection
^^^^^^^^^^^^^^^^^^

The system implements several layers of brute force protection:

* Failed Login Tracking: Tracks the number of failed login attempts per user
* Account Lockout: Requires CAPTCHA verification after 3 failed attempts within 5 minutes
* IP-based Rate Limiting: Limits the number of requests from an IP address
* CAPTCHA Integration: reCAPTCHA v2 integration for additional security

To set up CAPTCHA protection:

1. Sign up for reCAPTCHA at https://www.google.com/recaptcha
2. Add your site and get API keys
3. Configure the keys in config.json:

.. code-block:: json

    {
        "recaptcha_secret_key": "your-secret-key",
        "recaptcha_site_key": "your-site-key"
    }

4. Add the reCAPTCHA widget to your login form:

.. code-block:: html

    <script src="https://www.google.com/recaptcha/api.js"></script>
    <div class="g-recaptcha" data-sitekey="your-site-key"></div>

5. Pass the CAPTCHA response to the authentication call:

.. code-block:: python

    access_token, refresh_token = auth_manager.authenticate(
        username="user",
        password="password",
        captcha_response=g_recaptcha_response
    )

Token System
^^^^^^^^^^

The system uses a dual-token approach:

* Access Token: Short-lived token (default 1 hour) for API access
* Refresh Token: Long-lived token (default 24 hours) for obtaining new access tokens

Role Hierarchy
^^^^^^^^^^^^

The system implements a role hierarchy:

1. Admin: Full system access
2. Operator: Limited administrative access
3. Viewer: Read-only access

Error Handling
------------

The system defines several error types:

* ``AuthError``: Base authentication error
* ``InvalidCredentialsError``: Invalid username or password
* ``TokenError``: Invalid or expired token
* ``PermissionError``: Insufficient permissions

Example error handling:

.. code-block:: python

    try:
        auth_manager.check_permission(token, UserRole.ADMIN)
    except PermissionError:
        print("Insufficient permissions")
    except TokenError:
        print("Invalid or expired token") 