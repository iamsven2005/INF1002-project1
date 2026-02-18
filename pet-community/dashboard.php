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
$myPets = get_pets_for_user($userId);
$otherUsers = get_other_users($userId);
$allPets = get_pets();

$petsByUser = [];
foreach ($allPets as $pet) {
    $ownerId = (int) ($pet['user_id'] ?? 0);
    if (!isset($petsByUser[$ownerId])) {
        $petsByUser[$ownerId] = [];
    }
    $petsByUser[$ownerId][] = $pet;
}

render_header('Dashboard');
?>
<div class="row g-4">
    <div class="col-12">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <div class="d-flex flex-column flex-md-row gap-3 align-items-md-center justify-content-between">
                    <div>
                        <h1 class="h3 mb-1">Welcome, <?= e((string) ($user['full_name'] ?? $user['username'])) ?></h1>
                        <p class="text-secondary mb-0">This is your profile overview and the community directory.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <a href="edit_profile.php" class="btn btn-primary">Edit Profile</a>
                        <a href="delete_profile.php" class="btn btn-outline-danger">Delete Profile</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-12">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <h2 class="h4 mb-3">My Profile</h2>
                <div class="row g-3 align-items-start">
                    <div class="col-md-3">
                        <img src="<?= e((string) ($user['profile_photo'] ?? '')) ?>" alt="My profile photo" class="profile-thumb rounded-circle border">
                    </div>
                    <div class="col-md-9">
                        <p class="mb-1"><strong>Username:</strong> <?= e((string) ($user['username'] ?? '')) ?></p>
                        <p class="mb-1"><strong>Name:</strong> <?= e((string) ($user['full_name'] ?? '')) ?></p>
                        <p class="mb-1"><strong>Email:</strong> <?= e((string) ($user['email'] ?? '')) ?></p>
                        <p class="mb-0"><strong>Phone:</strong> <?= e((string) ($user['phone'] ?? '')) ?></p>
                    </div>
                </div>
                <hr>
                <h3 class="h5 mb-3">My Pets</h3>
                <?php if (empty($myPets)): ?>
                    <p class="text-secondary mb-0">No pets added yet.</p>
                <?php else: ?>
                    <div class="row g-3">
                        <?php foreach ($myPets as $pet): ?>
                            <div class="col-md-6 col-xl-4">
                                <div class="border rounded p-3 h-100">
                                    <img src="<?= e((string) ($pet['photo'] ?? '')) ?>" alt="Pet photo" class="pet-thumb rounded border mb-2">
                                    <p class="mb-1"><strong><?= e((string) ($pet['pet_name'] ?? '')) ?></strong></p>
                                    <p class="mb-1 text-secondary"><?= e((string) ($pet['breed'] ?? '')) ?></p>
                                    <p class="mb-0">Age: <?= e((string) ($pet['age'] ?? '')) ?></p>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>
            </div>
        </div>
    </div>

    <div class="col-12">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <h2 class="h4 mb-3">Other Community Members</h2>
                <?php if (empty($otherUsers)): ?>
                    <p class="text-secondary mb-0">No other users yet.</p>
                <?php else: ?>
                    <div class="row g-3">
                        <?php foreach ($otherUsers as $member): ?>
                            <?php $memberPets = $petsByUser[(int) ($member['id'] ?? 0)] ?? []; ?>
                            <div class="col-lg-6">
                                <div class="border rounded p-3 h-100">
                                    <div class="d-flex gap-3">
                                        <img src="<?= e((string) ($member['profile_photo'] ?? '')) ?>" alt="Member profile photo" class="member-thumb rounded-circle border">
                                        <div>
                                            <p class="mb-1"><strong><?= e((string) ($member['full_name'] ?? '')) ?></strong></p>
                                            <p class="mb-1 text-secondary">@<?= e((string) ($member['username'] ?? '')) ?></p>
                                            <p class="mb-1"><?= e((string) ($member['email'] ?? '')) ?></p>
                                            <p class="mb-2"><?= e((string) ($member['phone'] ?? '')) ?></p>
                                            <p class="mb-0 small text-secondary">Pets: <?= count($memberPets) ?></p>
                                        </div>
                                    </div>
                                    <?php if (!empty($memberPets)): ?>
                                        <div class="d-flex flex-wrap gap-2 mt-3">
                                            <?php foreach ($memberPets as $pet): ?>
                                                <span class="badge rounded-pill text-bg-light border"><?= e((string) ($pet['pet_name'] ?? 'Pet')) ?></span>
                                            <?php endforeach; ?>
                                        </div>
                                    <?php endif; ?>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>
            </div>
        </div>
    </div>
</div>
<?php render_footer(); ?>
