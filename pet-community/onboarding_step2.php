<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';
require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

require_wizard_step(2);

$data = get_onboarding_data();
$fullName = (string) ($data['full_name'] ?? '');
$email = (string) ($data['email'] ?? '');
$phone = (string) ($data['phone'] ?? '');
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $fullName = trim((string) ($_POST['full_name'] ?? ''));
    $email = trim((string) ($_POST['email'] ?? ''));
    $phone = trim((string) ($_POST['phone'] ?? ''));

    if ($fullName === '') {
        $error = 'Name is required.';
    } elseif (!is_valid_email($email)) {
        $error = 'Please enter a valid email address.';
    } elseif (!is_valid_phone($phone)) {
        $error = 'Please enter a valid phone number.';
    } else {
        update_onboarding_data([
            'full_name' => $fullName,
            'email' => $email,
            'phone' => $phone,
        ]);
        redirect('onboarding_step3.php');
    }
}

render_header('Onboarding - Step 2');
?>
<div class="row justify-content-center">
    <div class="col-lg-7">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <p class="step-label mb-2">Step 2 of 5</p>
                <h1 class="h4 mb-3">Personal Information</h1>

                <?php if ($error !== ''): ?>
                    <div class="alert alert-danger"><?= e($error) ?></div>
                <?php endif; ?>

                <form method="post" novalidate>
                    <div class="mb-3">
                        <label for="full_name" class="form-label">Full Name</label>
                        <input type="text" class="form-control" id="full_name" name="full_name" value="<?= e($fullName) ?>" required>
                    </div>
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" value="<?= e($email) ?>" required>
                    </div>
                    <div class="mb-4">
                        <label for="phone" class="form-label">Phone</label>
                        <input type="text" class="form-control" id="phone" name="phone" value="<?= e($phone) ?>" required>
                    </div>
                    <div class="d-flex justify-content-between">
                        <a class="btn btn-outline-secondary" href="onboarding_step1.php">Previous</a>
                        <button type="submit" class="btn btn-primary">Next</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
