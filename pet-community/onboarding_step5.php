<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';
require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

require_wizard_step(5);

$data = get_onboarding_data();
$pets = isset($data['pets']) && is_array($data['pets']) ? $data['pets'] : [];

render_header('Onboarding - Step 5');
?>
<div class="row justify-content-center">
    <div class="col-xl-9">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <p class="step-label mb-2">Step 5 of 5</p>
                <h1 class="h4 mb-3">Confirm and Save</h1>

                <div class="row g-4">
                    <div class="col-lg-6">
                        <div class="border rounded p-3 h-100">
                            <h2 class="h6 mb-3">Account</h2>
                            <p class="mb-1"><strong>Username:</strong> <?= e((string) ($data['username'] ?? '')) ?></p>
                            <p class="mb-0"><strong>Password:</strong> ********</p>
                        </div>
                    </div>
                    <div class="col-lg-6">
                        <div class="border rounded p-3 h-100">
                            <h2 class="h6 mb-3">Personal Info</h2>
                            <p class="mb-1"><strong>Name:</strong> <?= e((string) ($data['full_name'] ?? '')) ?></p>
                            <p class="mb-1"><strong>Email:</strong> <?= e((string) ($data['email'] ?? '')) ?></p>
                            <p class="mb-0"><strong>Phone:</strong> <?= e((string) ($data['phone'] ?? '')) ?></p>
                        </div>
                    </div>
                </div>

                <div class="border rounded p-3 mt-3">
                    <h2 class="h6 mb-3">Profile Photo</h2>
                    <?php if (!empty($data['profile_photo'])): ?>
                        <img src="<?= e((string) $data['profile_photo']) ?>" alt="Profile preview" class="profile-thumb rounded-circle border">
                    <?php endif; ?>
                </div>

                <div class="border rounded p-3 mt-3">
                    <h2 class="h6 mb-3">Pets</h2>
                    <div class="row g-3">
                        <?php foreach ($pets as $pet): ?>
                            <div class="col-md-6">
                                <div class="border rounded p-3 h-100">
                                    <div class="d-flex gap-3">
                                        <img src="<?= e((string) ($pet['photo'] ?? '')) ?>" alt="Pet photo" class="pet-thumb rounded border">
                                        <div>
                                            <p class="mb-1"><strong>Name:</strong> <?= e((string) ($pet['pet_name'] ?? '')) ?></p>
                                            <p class="mb-1"><strong>Breed:</strong> <?= e((string) ($pet['breed'] ?? '')) ?></p>
                                            <p class="mb-0"><strong>Age:</strong> <?= e((string) ($pet['age'] ?? '')) ?></p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    </div>
                </div>

                <div class="d-flex justify-content-between mt-4">
                    <a class="btn btn-outline-secondary" href="onboarding_step4.php">Previous</a>
                    <form method="post" action="save_onboarding.php">
                        <button type="submit" class="btn btn-success">Save Profile</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
