<?php
declare(strict_types=1);

if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

define('APP_NAME', 'Pet Lovers Community');
define('BASE_DIR', dirname(__DIR__));
define('DATA_DIR', BASE_DIR . DIRECTORY_SEPARATOR . 'data');
define('UPLOAD_DIR', BASE_DIR . DIRECTORY_SEPARATOR . 'uploads');
define('PROFILE_UPLOAD_DIR', UPLOAD_DIR . DIRECTORY_SEPARATOR . 'profiles');
define('PET_UPLOAD_DIR', UPLOAD_DIR . DIRECTORY_SEPARATOR . 'pets');
define('USERS_CSV', DATA_DIR . DIRECTORY_SEPARATOR . 'users.csv');
define('PETS_CSV', DATA_DIR . DIRECTORY_SEPARATOR . 'pets.csv');

define('USER_FIELDS', [
    'id',
    'username',
    'password_hash',
    'full_name',
    'email',
    'phone',
    'profile_photo',
    'created_at',
]);

define('PET_FIELDS', [
    'id',
    'user_id',
    'pet_name',
    'breed',
    'age',
    'photo',
]);

function ensure_storage(): void
{
    foreach ([DATA_DIR, UPLOAD_DIR, PROFILE_UPLOAD_DIR, PET_UPLOAD_DIR] as $directory) {
        if (!is_dir($directory)) {
            mkdir($directory, 0775, true);
        }
    }

    if (!file_exists(USERS_CSV)) {
        $userFile = fopen(USERS_CSV, 'wb');
        if ($userFile !== false) {
            fputcsv($userFile, USER_FIELDS);
            fclose($userFile);
        }
    }

    if (!file_exists(PETS_CSV)) {
        $petFile = fopen(PETS_CSV, 'wb');
        if ($petFile !== false) {
            fputcsv($petFile, PET_FIELDS);
            fclose($petFile);
        }
    }
}

ensure_storage();
