#include "BBGameMode.h"
#include "Actors/BlockActor.h"
#include "Actors/Projectile.h"
#include "Actors/Coin.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"

ABBGameMode::ABBGameMode()
{
    PrimaryActorTick.bCanEverTick = true;
    CurrentWave = 1;
    BlocksDestroyed = 0;
    SpawnInterval = 2.0f;
    DifficultyInterval = 30.0f;
}

void ABBGameMode::BeginPlay()
{
    Super::BeginPlay();
    SetupTimers();
}

void ABBGameMode::StartPlay()
{
    Super::StartPlay();
    SpawnBlockWave();
}

void ABBGameMode::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void ABBGameMode::SetupTimers()
{
    GetWorldTimerManager().SetTimer(SpawnTimerHandle, this, &ABBGameMode::SpawnBlockWave, SpawnInterval, true);
    GetWorldTimerManager().SetTimer(DifficultyTimerHandle, this, &ABBGameMode::IncreaseDifficulty, DifficultyInterval, true);
}

void ABBGameMode::SpawnBlockWave()
{
    if (!BlockClass) return;
    
    UWorld* World = GetWorld();
    if (!World) return;
    
    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
    
    for (int32 i = 0; i < 5 + CurrentWave; i++)
    {
        FVector SpawnLocation = FVector(
            FMath::RandRange(-800.0f, 800.0f),
            FMath::RandRange(-800.0f, 800.0f),
            1000.0f
        );
        
        FRotator SpawnRotation = FRotator::ZeroRotator;
        
        ABlockActor* SpawnedBlock = World->SpawnActor<ABlockActor>(BlockClass, SpawnLocation, SpawnRotation, SpawnParams);
        if (SpawnedBlock)
        {
            SpawnedBlock->SetHealth(100 * CurrentWave);
        }
    }
}

void ABBGameMode::IncreaseDifficulty()
{
    CurrentWave++;
    SpawnInterval = FMath::Max(0.5f, SpawnInterval - 0.1f);
    GetWorldTimerManager().SetTimer(SpawnTimerHandle, this, &ABBGameMode::SpawnBlockWave, SpawnInterval, true);
}
