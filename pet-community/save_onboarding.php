<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    redirect('onboarding_step5.php');
}

require_wizard_step(5);
$data = get_onboarding_data();

$username = trim((string) ($data['username'] ?? ''));
$password = (string) ($data['password'] ?? '');
$fullName = trim((string) ($data['full_name'] ?? ''));
$email = trim((string) ($data['email'] ?? ''));
$phone = trim((string) ($data['phone'] ?? ''));
$profilePhoto = (string) ($data['profile_photo'] ?? '');
$pets = isset($data['pets']) && is_array($data['pets']) ? $data['pets'] : [];

if ($username === '' || $password === '' || empty($pets)) {
    set_flash('danger', 'Your onboarding data is incomplete. Please finish all steps.');
    redirect('onboarding_step1.php');
}

if (find_user_by_username($username) !== null) {
    set_flash('danger', 'That username is already taken. Please pick another username.');
    redirect('onboarding_step1.php');
}

try {
    $newUser = create_user([
        'username' => $username,
        'password_hash' => password_hash($password, PASSWORD_DEFAULT),
        'full_name' => $fullName,
        'email' => $email,
        'phone' => $phone,
        'profile_photo' => $profilePhoto,
    ]);

    create_pets_for_user((int) $newUser['id'], $pets);
    clear_onboarding_data();
    login_user($newUser);

    set_flash('success', 'Profile created successfully.');
    redirect('dashboard.php');
} catch (Throwable $exception) {
    set_flash('danger', 'Unable to save your profile. Please try again.');
    redirect('onboarding_step5.php');
}
