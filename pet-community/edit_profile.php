<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/layout.php';

require_login();

$user = current_user();
if ($user === null) {
    logout_user();
    set_flash('warning', 'Your session expired. Please log in again.');
    redirect('login.php');
}

$userId = (int) $user['id'];
$persistedProfilePhoto = (string) ($user['profile_photo'] ?? '');
$profilePhotoPath = $persistedProfilePhoto;
$fullName = (string) ($user['full_name'] ?? '');
$email = (string) ($user['email'] ?? '');
$phone = (string) ($user['phone'] ?? '');

$currentPets = get_pets_for_user($userId);
$petsForForm = [];
foreach ($currentPets as $pet) {
    $petsForForm[] = [
        'pet_name' => (string) ($pet['pet_name'] ?? ''),
        'breed' => (string) ($pet['breed'] ?? ''),
        'age' => (string) ($pet['age'] ?? ''),
        'photo' => (string) ($pet['photo'] ?? ''),
    ];
}

if (empty($petsForForm)) {
    $petsForForm = [
        ['pet_name' => '', 'breed' => '', 'age' => '', 'photo' => ''],
    ];
}

$errors = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $fullName = trim((string) ($_POST['full_name'] ?? ''));
    $email = trim((string) ($_POST['email'] ?? ''));
    $phone = trim((string) ($_POST['phone'] ?? ''));
    $newPassword = (string) ($_POST['new_password'] ?? '');
    $profilePhotoPath = trim((string) ($_POST['existing_profile_photo'] ?? $persistedProfilePhoto));

    if ($fullName === '') {
        $errors[] = 'Name is required.';
    }
    if (!is_valid_email($email)) {
        $errors[] = 'Please enter a valid email address.';
    }
    if (!is_valid_phone($phone)) {
        $errors[] = 'Please enter a valid phone number.';
    }
    if ($newPassword !== '' && strlen($newPassword) < 6) {
        $errors[] = 'New password must be at least 6 characters.';
    }

    if (
        isset($_FILES['profile_photo']) &&
        (int) ($_FILES['profile_photo']['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_NO_FILE
    ) {
        try {
            $previousPath = $profilePhotoPath;
            $profilePhotoPath = store_uploaded_image(
                $_FILES['profile_photo'],
                PROFILE_UPLOAD_DIR,
                normalize_username((string) ($user['username'] ?? 'user')) . '_profile_edit'
            );

            if (
                $previousPath !== ''
                && $previousPath !== $persistedProfilePhoto
                && $previousPath !== $profilePhotoPath
            ) {
                remove_file_if_exists($previousPath);
            }
        } catch (Throwable $exception) {
            $errors[] = $exception->getMessage();
        }
    }

    $petResult = process_pet_form_data(
        $_POST,
        isset($_FILES['pet_photo']) && is_array($_FILES['pet_photo']) ? $_FILES['pet_photo'] : [],
        normalize_username((string) ($user['username'] ?? 'user')) . '_pet_edit'
    );
    $errors = array_merge($errors, $petResult['errors']);
    $petsForForm = !empty($petResult['draft_rows'])
        ? $petResult['draft_rows']
        : [['pet_name' => '', 'breed' => '', 'age' => '', 'photo' => '']];

    if (count($petResult['pets']) === 0) {
        $errors[] = 'Please add at least one pet with complete details.';
    }

    if (empty($errors)) {
        $updates = [
            'full_name' => $fullName,
            'email' => $email,
            'phone' => $phone,
            'profile_photo' => $profilePhotoPath,
        ];

        if ($newPassword !== '') {
            $updates['password_hash'] = password_hash($newPassword, PASSWORD_DEFAULT);
        }

        try {
            $updatedUser = update_user($userId, $updates);
            if ($updatedUser === null) {
                throw new RuntimeException('User profile could not be updated.');
            }

            replace_user_pets($userId, $petResult['pets']);

            $newPetPhotoPaths = array_map(
                static fn (array $pet): string => (string) ($pet['photo'] ?? ''),
                $petResult['pets']
            );
            foreach ($currentPets as $oldPet) {
                $oldPath = (string) ($oldPet['photo'] ?? '');
                if ($oldPath !== '' && !in_array($oldPath, $newPetPhotoPaths, true)) {
                    remove_file_if_exists($oldPath);
                }
            }

            if ($persistedProfilePhoto !== '' && $persistedProfilePhoto !== $profilePhotoPath) {
                remove_file_if_exists($persistedProfilePhoto);
            }

            set_flash('success', 'Profile updated successfully.');
            redirect('dashboard.php');
        } catch (Throwable $exception) {
            $errors[] = 'Unable to update profile right now. Please try again.';
        }
    }
}

render_header('Edit Profile');
?>
<div class="row justify-content-center">
    <div class="col-xl-9">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <h1 class="h4 mb-3">Edit Profile</h1>

                <?php if (!empty($errors)): ?>
                    <div class="alert alert-danger">
                        <ul class="mb-0">
                            <?php foreach ($errors as $error): ?>
                                <li><?= e($error) ?></li>
                            <?php endforeach; ?>
                        </ul>
                    </div>
                <?php endif; ?>

                <form method="post" enctype="multipart/form-data" novalidate>
                    <h2 class="h6 mt-2">Account</h2>
                    <div class="row g-3 mb-3">
                        <div class="col-md-6">
                            <label class="form-label" for="username">Username</label>
                            <input type="text" id="username" class="form-control" value="<?= e((string) ($user['username'] ?? '')) ?>" disabled>
                            <div class="form-text">Username cannot be changed.</div>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label" for="new_password">New Password (optional)</label>
                            <input type="password" id="new_password" name="new_password" class="form-control" minlength="6">
                        </div>
                    </div>

                    <h2 class="h6">Personal Info</h2>
                    <div class="row g-3 mb-3">
                        <div class="col-md-4">
                            <label for="full_name" class="form-label">Full Name</label>
                            <input type="text" class="form-control" id="full_name" name="full_name" value="<?= e($fullName) ?>" required>
                        </div>
                        <div class="col-md-4">
                            <label for="email" class="form-label">Email</label>
                            <input type="email" class="form-control" id="email" name="email" value="<?= e($email) ?>" required>
                        </div>
                        <div class="col-md-4">
                            <label for="phone" class="form-label">Phone</label>
                            <input type="text" class="form-control" id="phone" name="phone" value="<?= e($phone) ?>" required>
                        </div>
                    </div>

                    <h2 class="h6">Profile Photo</h2>
                    <div class="row g-3 align-items-end mb-3">
                        <div class="col-md-4">
                            <?php if ($profilePhotoPath !== ''): ?>
                                <img src="<?= e($profilePhotoPath) ?>" alt="Profile photo" class="profile-thumb rounded-circle border">
                            <?php endif; ?>
                        </div>
                        <div class="col-md-8">
                            <input type="hidden" name="existing_profile_photo" value="<?= e($profilePhotoPath) ?>">
                            <label for="profile_photo" class="form-label">Replace Profile Photo</label>
                            <input type="file" class="form-control" id="profile_photo" name="profile_photo" accept="image/*">
                        </div>
                    </div>

                    <h2 class="h6">Pet Information</h2>
                    <div id="pet-rows">
                        <?php foreach ($petsForForm as $pet): ?>
                            <div class="card mb-3 pet-row">
                                <div class="card-body">
                                    <div class="row g-3">
                                        <div class="col-md-4">
                                            <label class="form-label">Pet Name</label>
                                            <input type="text" class="form-control" name="pet_name[]" value="<?= e((string) ($pet['pet_name'] ?? '')) ?>">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label">Breed</label>
                                            <input type="text" class="form-control" name="pet_breed[]" value="<?= e((string) ($pet['breed'] ?? '')) ?>">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label">Age</label>
                                            <input type="number" min="0" max="50" class="form-control" name="pet_age[]" value="<?= e((string) ($pet['age'] ?? '')) ?>">
                                        </div>
                                        <div class="col-md-8">
                                            <label class="form-label">Pet Photo</label>
                                            <input type="hidden" name="existing_pet_photo[]" value="<?= e((string) ($pet['photo'] ?? '')) ?>">
                                            <input type="file" class="form-control" name="pet_photo[]" accept="image/*">
                                        </div>
                                        <div class="col-md-4 d-flex align-items-end justify-content-between gap-2">
                                            <?php if (!empty($pet['photo'])): ?>
                                                <img src="<?= e((string) $pet['photo']) ?>" alt="Pet photo preview" class="pet-thumb rounded border">
                                            <?php else: ?>
                                                <span class="text-secondary small">No photo selected</span>
                                            <?php endif; ?>
                                            <button type="button" class="btn btn-outline-danger btn-sm remove-pet">Remove</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    </div>

                    <button type="button" class="btn btn-outline-primary mb-4" id="add-pet">Add Another Pet</button>

                    <div class="d-flex justify-content-between">
                        <a href="dashboard.php" class="btn btn-outline-secondary">Cancel</a>
                        <button type="submit" class="btn btn-primary">Save Changes</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>

<template id="pet-row-template">
    <div class="card mb-3 pet-row">
        <div class="card-body">
            <div class="row g-3">
                <div class="col-md-4">
                    <label class="form-label">Pet Name</label>
                    <input type="text" class="form-control" name="pet_name[]">
                </div>
                <div class="col-md-4">
                    <label class="form-label">Breed</label>
                    <input type="text" class="form-control" name="pet_breed[]">
                </div>
                <div class="col-md-4">
                    <label class="form-label">Age</label>
                    <input type="number" min="0" max="50" class="form-control" name="pet_age[]">
                </div>
                <div class="col-md-8">
                    <label class="form-label">Pet Photo</label>
                    <input type="hidden" name="existing_pet_photo[]" value="">
                    <input type="file" class="form-control" name="pet_photo[]" accept="image/*">
                </div>
                <div class="col-md-4 d-flex align-items-end justify-content-between gap-2">
                    <span class="text-secondary small">No photo selected</span>
                    <button type="button" class="btn btn-outline-danger btn-sm remove-pet">Remove</button>
                </div>
            </div>
        </div>
    </div>
</template>
<?php render_footer(['assets/js/pets.js']); ?>
