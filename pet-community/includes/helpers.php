<?php
declare(strict_types=1);

require_once __DIR__ . '/config.php';

function redirect(string $path): never
{
    header("Location: {$path}");
    exit;
}

function e(string $value): string
{
    return htmlspecialchars($value, ENT_QUOTES, 'UTF-8');
}

function normalize_username(string $username): string
{
    return strtolower(trim($username));
}

function set_flash(string $type, string $message): void
{
    $_SESSION['flash'] = [
        'type' => $type,
        'message' => $message,
    ];
}

function get_flash(): ?array
{
    if (!isset($_SESSION['flash']) || !is_array($_SESSION['flash'])) {
        return null;
    }

    $flash = $_SESSION['flash'];
    unset($_SESSION['flash']);

    return $flash;
}

function is_valid_email(string $email): bool
{
    return filter_var($email, FILTER_VALIDATE_EMAIL) !== false;
}

function is_valid_phone(string $phone): bool
{
    return (bool) preg_match('/^[0-9+\-\s()]{7,20}$/', $phone);
}

function detect_uploaded_image_extension(string $tmpFile): ?string
{
    $allowedMimes = [
        'image/jpeg' => 'jpg',
        'image/png' => 'png',
        'image/gif' => 'gif',
        'image/webp' => 'webp',
    ];

    if (function_exists('finfo_open')) {
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        if ($finfo !== false) {
            $mimeType = finfo_file($finfo, $tmpFile);
            finfo_close($finfo);
            if (is_string($mimeType) && isset($allowedMimes[$mimeType])) {
                return $allowedMimes[$mimeType];
            }
        }
    }

    if (function_exists('mime_content_type')) {
        $mimeType = mime_content_type($tmpFile);
        if (is_string($mimeType) && isset($allowedMimes[$mimeType])) {
            return $allowedMimes[$mimeType];
        }
    }

    if (function_exists('exif_imagetype')) {
        $typeMap = [
            IMAGETYPE_JPEG => 'jpg',
            IMAGETYPE_PNG => 'png',
            IMAGETYPE_GIF => 'gif',
        ];

        if (defined('IMAGETYPE_WEBP')) {
            $typeMap[(int) constant('IMAGETYPE_WEBP')] = 'webp';
        }

        $imageType = @exif_imagetype($tmpFile);
        if (is_int($imageType) && isset($typeMap[$imageType])) {
            return $typeMap[$imageType];
        }
    }

    if (function_exists('getimagesize')) {
        $imageInfo = @getimagesize($tmpFile);
        if (is_array($imageInfo)) {
            $mimeType = (string) ($imageInfo['mime'] ?? '');
            if (isset($allowedMimes[$mimeType])) {
                return $allowedMimes[$mimeType];
            }
        }
    }

    return null;
}

function store_uploaded_image(array $file, string $destinationDir, string $prefix): string
{
    $errorCode = (int) ($file['error'] ?? UPLOAD_ERR_NO_FILE);
    if ($errorCode !== UPLOAD_ERR_OK) {
        throw new RuntimeException('Please upload a valid image file.');
    }

    $tmpFile = (string) ($file['tmp_name'] ?? '');
    if ($tmpFile === '' || !is_uploaded_file($tmpFile)) {
        throw new RuntimeException('The uploaded file could not be processed.');
    }

    $fileSize = (int) ($file['size'] ?? 0);
    if ($fileSize <= 0 || $fileSize > 4 * 1024 * 1024) {
        throw new RuntimeException('Image size must be between 1 byte and 4 MB.');
    }

    $extension = detect_uploaded_image_extension($tmpFile);
    if ($extension === null) {
        throw new RuntimeException('Allowed image types: JPG, PNG, GIF, WEBP.');
    }

    $cleanPrefix = preg_replace('/[^a-zA-Z0-9_-]/', '', $prefix) ?: 'img';
    $fileName = sprintf('%s_%s.%s', $cleanPrefix, bin2hex(random_bytes(8)), $extension);
    $destinationPath = rtrim($destinationDir, '/\\') . DIRECTORY_SEPARATOR . $fileName;

    if (!move_uploaded_file($tmpFile, $destinationPath)) {
        throw new RuntimeException('Failed to save uploaded image.');
    }

    $relativePath = substr($destinationPath, strlen(BASE_DIR) + 1);

    return str_replace('\\', '/', $relativePath);
}

function remove_file_if_exists(string $relativePath): void
{
    $cleanPath = ltrim($relativePath, '/\\');
    if ($cleanPath === '') {
        return;
    }

    $absolutePath = BASE_DIR . DIRECTORY_SEPARATOR . str_replace(['/', '\\'], DIRECTORY_SEPARATOR, $cleanPath);
    if (is_file($absolutePath)) {
        @unlink($absolutePath);
    }
}

function process_pet_form_data(array $postData, array $fileData, string $photoPrefix): array
{
    $petNames = isset($postData['pet_name']) && is_array($postData['pet_name']) ? $postData['pet_name'] : [];
    $petBreeds = isset($postData['pet_breed']) && is_array($postData['pet_breed']) ? $postData['pet_breed'] : [];
    $petAges = isset($postData['pet_age']) && is_array($postData['pet_age']) ? $postData['pet_age'] : [];
    $existingPhotos = isset($postData['existing_pet_photo']) && is_array($postData['existing_pet_photo']) ? $postData['existing_pet_photo'] : [];

    $fileNames = isset($fileData['name']) && is_array($fileData['name']) ? $fileData['name'] : [];
    $fileErrors = isset($fileData['error']) && is_array($fileData['error']) ? $fileData['error'] : [];
    $fileTmpNames = isset($fileData['tmp_name']) && is_array($fileData['tmp_name']) ? $fileData['tmp_name'] : [];
    $fileSizes = isset($fileData['size']) && is_array($fileData['size']) ? $fileData['size'] : [];
    $fileTypes = isset($fileData['type']) && is_array($fileData['type']) ? $fileData['type'] : [];

    $totalRows = max(
        count($petNames),
        count($petBreeds),
        count($petAges),
        count($existingPhotos),
        count($fileNames)
    );

    $pets = [];
    $draftRows = [];
    $errors = [];

    for ($index = 0; $index < $totalRows; $index++) {
        $petNumber = $index + 1;
        $name = trim((string) ($petNames[$index] ?? ''));
        $breed = trim((string) ($petBreeds[$index] ?? ''));
        $ageRaw = trim((string) ($petAges[$index] ?? ''));
        $photoPath = trim((string) ($existingPhotos[$index] ?? ''));
        $fileError = isset($fileErrors[$index]) ? (int) $fileErrors[$index] : UPLOAD_ERR_NO_FILE;
        $hasUpload = $fileError !== UPLOAD_ERR_NO_FILE;

        $hasAnyData = $name !== ''
            || $breed !== ''
            || $ageRaw !== ''
            || $photoPath !== ''
            || $hasUpload;

        if (!$hasAnyData) {
            continue;
        }

        if ($hasUpload) {
            if ($fileError !== UPLOAD_ERR_OK) {
                $errors[] = "Pet #{$petNumber}: upload failed.";
            } else {
                $singleFile = [
                    'name' => (string) ($fileNames[$index] ?? ''),
                    'type' => (string) ($fileTypes[$index] ?? ''),
                    'tmp_name' => (string) ($fileTmpNames[$index] ?? ''),
                    'error' => $fileError,
                    'size' => (int) ($fileSizes[$index] ?? 0),
                ];

                try {
                    $photoPath = store_uploaded_image($singleFile, PET_UPLOAD_DIR, "{$photoPrefix}_{$petNumber}");
                } catch (RuntimeException $exception) {
                    $errors[] = "Pet #{$petNumber}: {$exception->getMessage()}";
                }
            }
        }

        $rowErrors = [];
        if ($name === '') {
            $rowErrors[] = 'name is required';
        }
        if ($breed === '') {
            $rowErrors[] = 'breed is required';
        }

        $ageValue = -1;
        if ($ageRaw === '' || !ctype_digit($ageRaw)) {
            $rowErrors[] = 'age must be a whole number';
        } else {
            $ageValue = (int) $ageRaw;
            if ($ageValue < 0 || $ageValue > 50) {
                $rowErrors[] = 'age must be between 0 and 50';
            }
        }

        if ($photoPath === '') {
            $rowErrors[] = 'photo is required';
        }

        $draftRows[] = [
            'pet_name' => $name,
            'breed' => $breed,
            'age' => $ageRaw,
            'photo' => $photoPath,
        ];

        if (!empty($rowErrors)) {
            $errors[] = "Pet #{$petNumber}: " . implode(', ', $rowErrors) . '.';
            continue;
        }

        $pets[] = [
            'pet_name' => $name,
            'breed' => $breed,
            'age' => (string) $ageValue,
            'photo' => $photoPath,
        ];
    }

    return [
        'pets' => $pets,
        'draft_rows' => $draftRows,
        'errors' => $errors,
    ];
}
