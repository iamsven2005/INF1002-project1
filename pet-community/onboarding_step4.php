<?php
declare(strict_types=1);

require_once __DIR__ . '/includes/wizard.php';
require_once __DIR__ . '/includes/layout.php';

if (is_logged_in()) {
    redirect('dashboard.php');
}

require_wizard_step(4);

$data = get_onboarding_data();
$petPrefix = normalize_username((string) ($data['username'] ?? 'user')) . '_pet';
$errors = [];
$petsForForm = isset($data['pets']) && is_array($data['pets']) ? $data['pets'] : [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $result = process_pet_form_data(
        $_POST,
        isset($_FILES['pet_photo']) && is_array($_FILES['pet_photo']) ? $_FILES['pet_photo'] : [],
        $petPrefix
    );

    $errors = $result['errors'];
    $petsForForm = $result['draft_rows'];

    if (count($result['pets']) === 0) {
        $errors[] = 'Please add at least one pet with complete details.';
    }

    if (empty($errors)) {
        update_onboarding_data(['pets' => $result['pets']]);
        redirect('onboarding_step5.php');
    }
}

if (empty($petsForForm)) {
    $petsForForm = [
        [
            'pet_name' => '',
            'breed' => '',
            'age' => '',
            'photo' => '',
        ],
    ];
}

render_header('Onboarding - Step 4');
?>
<div class="row justify-content-center">
    <div class="col-xl-9">
        <div class="card border-0 shadow-sm">
            <div class="card-body p-4">
                <p class="step-label mb-2">Step 4 of 5</p>
                <h1 class="h4 mb-3">Pet Information</h1>

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
                        <a class="btn btn-outline-secondary" href="onboarding_step3.php">Previous</a>
                        <button type="submit" class="btn btn-primary">Next</button>
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
