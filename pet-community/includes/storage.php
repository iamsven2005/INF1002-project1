<?php
declare(strict_types=1);

require_once __DIR__ . '/helpers.php';

function csv_read_all(string $filePath): array
{
    $handle = fopen($filePath, 'cb+');
    if ($handle === false) {
        return [];
    }

    flock($handle, LOCK_SH);
    rewind($handle);

    $headers = fgetcsv($handle);
    if ($headers === false || $headers === [null]) {
        flock($handle, LOCK_UN);
        fclose($handle);
        return [];
    }

    $rows = [];
    while (($columns = fgetcsv($handle)) !== false) {
        if ($columns === [null]) {
            continue;
        }

        $row = [];
        foreach ($headers as $index => $header) {
            $row[$header] = (string) ($columns[$index] ?? '');
        }
        $rows[] = $row;
    }

    flock($handle, LOCK_UN);
    fclose($handle);

    return $rows;
}

function csv_write_all(string $filePath, array $headers, array $rows): void
{
    $handle = fopen($filePath, 'cb+');
    if ($handle === false) {
        throw new RuntimeException('Unable to write CSV file.');
    }

    flock($handle, LOCK_EX);
    ftruncate($handle, 0);
    rewind($handle);

    fputcsv($handle, $headers);
    foreach ($rows as $row) {
        $line = [];
        foreach ($headers as $header) {
            $line[] = (string) ($row[$header] ?? '');
        }
        fputcsv($handle, $line);
    }

    fflush($handle);
    flock($handle, LOCK_UN);
    fclose($handle);
}

function csv_next_id(array $rows): int
{
    $maxId = 0;
    foreach ($rows as $row) {
        $id = (int) ($row['id'] ?? 0);
        if ($id > $maxId) {
            $maxId = $id;
        }
    }

    return $maxId + 1;
}

function get_users(): array
{
    return csv_read_all(USERS_CSV);
}

function get_pets(): array
{
    return csv_read_all(PETS_CSV);
}

function find_user_by_username(string $username): ?array
{
    $target = normalize_username($username);
    foreach (get_users() as $user) {
        if (normalize_username((string) ($user['username'] ?? '')) === $target) {
            return $user;
        }
    }

    return null;
}

function find_user_by_id(int $userId): ?array
{
    foreach (get_users() as $user) {
        if ((int) ($user['id'] ?? 0) === $userId) {
            return $user;
        }
    }

    return null;
}

function create_user(array $data): array
{
    $users = get_users();
    $nextId = csv_next_id($users);

    $user = [
        'id' => (string) $nextId,
        'username' => (string) ($data['username'] ?? ''),
        'password_hash' => (string) ($data['password_hash'] ?? ''),
        'full_name' => (string) ($data['full_name'] ?? ''),
        'email' => (string) ($data['email'] ?? ''),
        'phone' => (string) ($data['phone'] ?? ''),
        'profile_photo' => (string) ($data['profile_photo'] ?? ''),
        'created_at' => date('c'),
    ];

    $users[] = $user;
    csv_write_all(USERS_CSV, USER_FIELDS, $users);

    return $user;
}

function update_user(int $userId, array $updates): ?array
{
    $users = get_users();
    $updatedUser = null;

    foreach ($users as &$user) {
        if ((int) ($user['id'] ?? 0) !== $userId) {
            continue;
        }

        $updatedUser = array_merge($user, $updates);
        $updatedUser['id'] = (string) $user['id'];
        $updatedUser['username'] = (string) $user['username'];
        $user = $updatedUser;
        break;
    }
    unset($user);

    if ($updatedUser === null) {
        return null;
    }

    csv_write_all(USERS_CSV, USER_FIELDS, $users);

    return $updatedUser;
}

function create_pets_for_user(int $userId, array $pets): void
{
    $allPets = get_pets();
    $nextId = csv_next_id($allPets);

    foreach ($pets as $pet) {
        $allPets[] = [
            'id' => (string) $nextId,
            'user_id' => (string) $userId,
            'pet_name' => (string) ($pet['pet_name'] ?? ''),
            'breed' => (string) ($pet['breed'] ?? ''),
            'age' => (string) ($pet['age'] ?? ''),
            'photo' => (string) ($pet['photo'] ?? ''),
        ];
        $nextId++;
    }

    csv_write_all(PETS_CSV, PET_FIELDS, $allPets);
}

function get_pets_for_user(int $userId): array
{
    $pets = [];
    foreach (get_pets() as $pet) {
        if ((int) ($pet['user_id'] ?? 0) === $userId) {
            $pets[] = $pet;
        }
    }

    return $pets;
}

function replace_user_pets(int $userId, array $pets): void
{
    $allPets = get_pets();
    $nextId = csv_next_id($allPets);
    $remainingPets = [];

    foreach ($allPets as $pet) {
        if ((int) ($pet['user_id'] ?? 0) !== $userId) {
            $remainingPets[] = $pet;
        }
    }

    foreach ($pets as $pet) {
        $remainingPets[] = [
            'id' => (string) $nextId,
            'user_id' => (string) $userId,
            'pet_name' => (string) ($pet['pet_name'] ?? ''),
            'breed' => (string) ($pet['breed'] ?? ''),
            'age' => (string) ($pet['age'] ?? ''),
            'photo' => (string) ($pet['photo'] ?? ''),
        ];
        $nextId++;
    }

    csv_write_all(PETS_CSV, PET_FIELDS, $remainingPets);
}

function delete_user_and_pets(int $userId): array
{
    $users = get_users();
    $remainingUsers = [];
    $deletedUser = null;

    foreach ($users as $user) {
        if ((int) ($user['id'] ?? 0) === $userId) {
            $deletedUser = $user;
            continue;
        }
        $remainingUsers[] = $user;
    }

    if ($deletedUser !== null) {
        csv_write_all(USERS_CSV, USER_FIELDS, $remainingUsers);
    }

    $pets = get_pets();
    $remainingPets = [];
    $deletedPets = [];

    foreach ($pets as $pet) {
        if ((int) ($pet['user_id'] ?? 0) === $userId) {
            $deletedPets[] = $pet;
            continue;
        }
        $remainingPets[] = $pet;
    }

    csv_write_all(PETS_CSV, PET_FIELDS, $remainingPets);

    return [
        'user' => $deletedUser,
        'pets' => $deletedPets,
    ];
}

function get_other_users(int $currentUserId): array
{
    $others = [];
    foreach (get_users() as $user) {
        if ((int) ($user['id'] ?? 0) !== $currentUserId) {
            $others[] = $user;
        }
    }

    usort(
        $others,
        static fn (array $a, array $b): int => strcmp(
            (string) ($a['username'] ?? ''),
            (string) ($b['username'] ?? '')
        )
    );

    return $others;
}
