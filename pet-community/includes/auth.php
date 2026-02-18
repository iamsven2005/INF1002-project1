<?php
declare(strict_types=1);

require_once __DIR__ . '/storage.php';

function login_user(array $user): void
{
    $_SESSION['user_id'] = (int) ($user['id'] ?? 0);
}

function logout_user(): void
{
    unset($_SESSION['user_id']);
    unset($_SESSION['onboarding']);
}

function current_user_id(): ?int
{
    if (!isset($_SESSION['user_id'])) {
        return null;
    }

    return (int) $_SESSION['user_id'];
}

function is_logged_in(): bool
{
    $userId = current_user_id();
    if ($userId === null) {
        return false;
    }

    return find_user_by_id($userId) !== null;
}

function current_user(): ?array
{
    $userId = current_user_id();
    if ($userId === null) {
        return null;
    }

    return find_user_by_id($userId);
}

function require_login(): void
{
    if (!is_logged_in()) {
        set_flash('warning', 'Please log in first.');
        redirect('login.php');
    }
}
