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

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $result = delete_user_and_pets((int) $user['id']);

    $deletedUser = $result['user'];
    $deletedPets = $result['pets'];

    if (is_array($deletedUser)) {
        $profilePhoto = (string) ($deletedUser['profile_photo'] ?? '');
        if ($profilePhoto !== '') {
            remove_file_if_exists($profilePhoto);
        }
    }

    if (is_array($deletedPets)) {
        foreach ($deletedPets as $pet) {
            $photoPath = (string) ($pet['photo'] ?? '');
            if ($photoPath !== '') {
                remove_file_if_exists($photoPath);
            }
        }
    }

    logout_user();
    set_flash('info', 'Your profile has been deleted.');
    redirect('index.php');
}

render_header('Delete Profile');
?>
<div class="row justify-content-center">
    <div class="col-md-8 col-lg-6">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <h1 class="h4 text-danger mb-3">Delete Profile</h1>
                <p class="mb-4">
                    This action is permanent. Your account and pet data will be removed from the community.
                </p>
                <form method="post">
                    <div class="d-flex justify-content-between">
                        <a href="dashboard.php" class="btn btn-outline-secondary">Cancel</a>
                        <button type="submit" class="btn btn-danger">Delete My Profile</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
