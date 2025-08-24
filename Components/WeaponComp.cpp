#include "WeaponComp.h"
#include "Projectile.h"
#include "Engine/World.h"
#include "TimerManager.h"

UWeaponComp::UWeaponComp()
{
    PrimaryComponentTick.bCanEverTick = true;
    FireRate = 0.1f;
    bCanFire = true;
    MuzzleOffset = FVector(100.0f, 0.0f, 0.0f);
}

void UWeaponComp::BeginPlay()
{
    Super::BeginPlay();
}

void UWeaponComp::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UWeaponComp::Fire()
{
    if (!bCanFire || !ProjectileClass) return;
    
    AActor* Owner = GetOwner();
    if (!Owner) return;
    
    UWorld* World = GetWorld();
    if (!World) return;
    
    FVector SpawnLocation = Owner->GetActorLocation() + Owner->GetActorRotation().RotateVector(MuzzleOffset);
    FRotator SpawnRotation = Owner->GetActorRotation();
    
    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = Owner;
    SpawnParams.Instigator = Owner->GetInstigator();
    
    AProjectile* Projectile = World->SpawnActor<AProjectile>(ProjectileClass, SpawnLocation, SpawnRotation, SpawnParams);
    if (Projectile)
    {
        bCanFire = false;
        World->GetTimerManager().SetTimer(TimerHandle_ShotTimerExpired, this, &UWeaponComp::ResetShot, FireRate);
    }
}

void UWeaponComp::ResetShot()
{
    bCanFire = true;
}
