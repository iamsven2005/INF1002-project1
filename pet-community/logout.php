<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/auth.php';

logout_user();
set_flash('info', 'You have been logged out.');
redirect('login.php');
