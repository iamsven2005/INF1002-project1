<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';
require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

$data = get_onboarding_data();
$username = (string) ($data['username'] ?? '');
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim((string) ($_POST['username'] ?? ''));
    $password = (string) ($_POST['password'] ?? '');
    $confirmPassword = (string) ($_POST['confirm_password'] ?? '');

    if (!preg_match('/^[A-Za-z0-9_]{3,20}$/', $username)) {
        $error = 'Username must be 3-20 characters and use only letters, numbers, or underscores.';
    } elseif (strlen($password) < 6) {
        $error = 'Password must be at least 6 characters.';
    } elseif ($password !== $confirmPassword) {
        $error = 'Password confirmation does not match.';
    } else {
        $existingUser = find_user_by_username($username);
        if ($existingUser !== null) {
            $error = 'That username is already taken. Please choose another.';
        } else {
            update_onboarding_data([
                'username' => $username,
                'password' => $password,
            ]);
            redirect('onboarding_step2.php');
        }
    }
}

render_header('Onboarding - Step 1');
?>
<div class="row justify-content-center">
    <div class="col-lg-7">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <p class="step-label mb-2">Step 1 of 5</p>
                <h1 class="h4 mb-3">Create Your Login</h1>

                <?php if ($error !== ''): ?>
                    <div class="alert alert-danger"><?= e($error) ?></div>
                <?php endif; ?>

                <form method="post" novalidate>
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <input type="text" class="form-control" id="username" name="username" value="<?= e($username) ?>" required>
                        <div class="form-text">This cannot be changed later.</div>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <div class="mb-4">
                        <label for="confirm_password" class="form-label">Confirm Password</label>
                        <input type="password" class="form-control" id="confirm_password" name="confirm_password" required>
                    </div>

                    <div class="d-flex justify-content-between">
                        <a class="btn btn-outline-secondary" href="index.php">Back</a>
                        <button type="submit" class="btn btn-primary">Next</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
