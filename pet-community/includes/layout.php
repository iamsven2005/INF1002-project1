<?php
declare(strict_types=1);

require_once __DIR__ . '/auth.php';

function render_header(string $title): void
{
    $user = current_user();
    $flash = get_flash();
    ?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title><?= e($title) ?> | <?= e(APP_NAME) ?></title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="assets/css/styles.css">
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark shadow-sm">
    <div class="container">
        <a class="navbar-brand fw-semibold" href="index.php"><?= e(APP_NAME) ?></a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#appNav" aria-controls="appNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="appNav">
            <ul class="navbar-nav ms-auto align-items-lg-center gap-lg-2">
                <?php if ($user !== null): ?>
                    <li class="nav-item"><a class="nav-link" href="dashboard.php">Dashboard</a></li>
                    <li class="nav-item"><a class="nav-link" href="edit_profile.php">Edit Profile</a></li>
                    <li class="nav-item"><a class="btn btn-outline-light btn-sm ms-lg-2" href="logout.php">Logout</a></li>
                <?php else: ?>
                    <li class="nav-item"><a class="nav-link" href="login.php">Login</a></li>
                    <li class="nav-item"><a class="btn btn-primary btn-sm ms-lg-2" href="onboarding_step1.php">Get Started</a></li>
                <?php endif; ?>
            </ul>
        </div>
    </div>
</nav>
<main class="container py-4 py-lg-5">
    <?php if ($flash !== null): ?>
        <div class="alert alert-<?= e((string) ($flash['type'] ?? 'info')) ?> alert-dismissible fade show" role="alert">
            <?= e((string) ($flash['message'] ?? '')) ?>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    <?php endif; ?>
    <?php
}

function render_footer(array $scripts = []): void
{
    ?>
</main>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<?php foreach ($scripts as $script): ?>
    <script src="<?= e($script) ?>"></script>
<?php endforeach; ?>
</body>
</html>
<?php
}
