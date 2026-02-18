<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';
require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

require_wizard_step(3);

$data = get_onboarding_data();
$photoPath = (string) ($data['profile_photo'] ?? '');
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        if (
            isset($_FILES['profile_photo']) &&
            (int) ($_FILES['profile_photo']['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_NO_FILE
        ) {
            $newPhotoPath = store_uploaded_image(
                $_FILES['profile_photo'],
                PROFILE_UPLOAD_DIR,
                normalize_username((string) ($data['username'] ?? 'user')) . '_profile'
            );

            if ($photoPath !== '' && $photoPath !== $newPhotoPath) {
                remove_file_if_exists($photoPath);
            }
            $photoPath = $newPhotoPath;
        }

        if ($photoPath === '') {
            throw new RuntimeException('Please upload a profile photo before continuing.');
        }

        update_onboarding_data(['profile_photo' => $photoPath]);
        redirect('onboarding_step4.php');
    } catch (Throwable $exception) {
        $error = $exception->getMessage();
    }
}

render_header('Onboarding - Step 3');
?>
<div class="row justify-content-center">
    <div class="col-lg-7">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <p class="step-label mb-2">Step 3 of 5</p>
                <h1 class="h4 mb-3">Upload Profile Photo</h1>

                <?php if ($error !== ''): ?>
                    <div class="alert alert-danger"><?= e($error) ?></div>
                <?php endif; ?>

                <form method="post" enctype="multipart/form-data" novalidate>
                    <?php if ($photoPath !== ''): ?>
                        <div class="mb-3">
                            <p class="form-label">Current Photo</p>
                            <img src="<?= e($photoPath) ?>" alt="Profile photo preview" class="profile-thumb rounded-circle border">
                        </div>
                    <?php endif; ?>
                    <div class="mb-4">
                        <label for="profile_photo" class="form-label">Choose Photo</label>
                        <input type="file" class="form-control" id="profile_photo" name="profile_photo" accept="image/*" <?= $photoPath === '' ? 'required' : '' ?>>
                    </div>
                    <div class="d-flex justify-content-between">
                        <a class="btn btn-outline-secondary" href="onboarding_step2.php">Previous</a>
                        <button type="submit" class="btn btn-primary">Next</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
