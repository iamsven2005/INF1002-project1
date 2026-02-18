<?php
declare(strict_types=1);

require_once __DIR__ . '/auth.php';

function get_onboarding_data(): array
{
    if (!isset($_SESSION['onboarding']) || !is_array($_SESSION['onboarding'])) {
        $_SESSION['onboarding'] = [
            'username' => '',
            'password' => '',
            'full_name' => '',
            'email' => '',
            'phone' => '',
            'profile_photo' => '',
            'pets' => [],
        ];
    }

    return $_SESSION['onboarding'];
}

function update_onboarding_data(array $updates): void
{
    $current = get_onboarding_data();
    $_SESSION['onboarding'] = array_merge($current, $updates);
}

function clear_onboarding_data(): void
{
    unset($_SESSION['onboarding']);
}

function wizard_step_complete(int $step, array $data): bool
{
    switch ($step) {
        case 1:
            return trim((string) ($data['username'] ?? '')) !== ''
                && trim((string) ($data['password'] ?? '')) !== '';
        case 2:
            return trim((string) ($data['full_name'] ?? '')) !== ''
                && trim((string) ($data['email'] ?? '')) !== ''
                && trim((string) ($data['phone'] ?? '')) !== '';
        case 3:
            return trim((string) ($data['profile_photo'] ?? '')) !== '';
        case 4:
            return isset($data['pets']) && is_array($data['pets']) && count($data['pets']) > 0;
        default:
            return false;
    }
}

function require_wizard_step(int $step): void
{
    $data = get_onboarding_data();

    for ($current = 1; $current < $step; $current++) {
        if (!wizard_step_complete($current, $data)) {
            redirect("onboarding_step{$current}.php");
        }
    }
}
