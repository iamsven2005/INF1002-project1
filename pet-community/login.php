<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

$username = '';
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim((string) ($_POST['username'] ?? ''));
    $password = (string) ($_POST['password'] ?? '');

    $user = find_user_by_username($username);
    if ($user === null || !password_verify($password, (string) ($user['password_hash'] ?? ''))) {
        $error = 'Invalid username or password.';
    } else {
        login_user($user);
        set_flash('success', 'Welcome back!');
        redirect('dashboard.php');
    }
}

render_header('Login');
?>
<div class="row justify-content-center">
    <div class="col-md-7 col-lg-5">
        <div class="card shadow-sm border-0">
            <div class="card-body p-4">
                <h1 class="h4 mb-3">Log In</h1>

                <?php if ($error !== ''): ?>
                    <div class="alert alert-danger"><?= e($error) ?></div>
                <?php endif; ?>

                <form method="post" novalidate>
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <input type="text" class="form-control" id="username" name="username" value="<?= e($username) ?>" required>
                    </div>
                    <div class="mb-4">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Login</button>
                </form>
                <p class="small text-secondary mt-3 mb-0">
                    New here? <a href="onboarding_step1.php">Start onboarding</a>
                </p>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
