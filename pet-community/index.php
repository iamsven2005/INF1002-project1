<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/layout.php';
require_once __DIR__ . '/includes/wizard.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

render_header('Welcome');
?>
<section class="p-4 p-lg-5 rounded-4 hero-panel">
    <div class="row align-items-center g-4">
        <div class="col-lg-7">
            <h1 class="display-6 fw-bold mb-3">Join a Friendly Community for Pet Lovers</h1>
            <p class="lead text-secondary mb-4">
                Complete a 5-step onboarding wizard to set up your account, add your pet details, and connect with other members.
            </p>
            <div class="d-flex flex-wrap gap-2">
                <a href="onboarding_step1.php" class="btn btn-primary btn-lg">Start Onboarding</a>
                <a href="login.php" class="btn btn-outline-dark btn-lg">I Already Have an Account</a>
            </div>
        </div>
        <div class="col-lg-5">
            <div class="card border-0 shadow-sm">
                <div class="card-body p-4">
                    <h2 class="h5">Onboarding Steps</h2>
                    <ol class="mb-0 ps-3">
                        <li>Username and Password</li>
                        <li>Personal Information</li>
                        <li>Profile Photo</li>
                        <li>Pet Information</li>
                        <li>Confirmation and Save</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>
</section>
<?php render_footer(); ?>
