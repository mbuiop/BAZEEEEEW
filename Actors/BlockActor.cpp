#include "BlockActor.h"
#include "Components/StaticMeshComponent.h"
#include "FighterChar.h"
#include "Projectile.h"
#include "Kismet/GameplayStatics.h"
#include "BBGameMode.h"

ABlockActor::ABlockActor()
{
    PrimaryActorTick.bCanEverTick = true;
    
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    RootComponent = MeshComponent;
    
    MeshComponent->SetSimulatePhysics(true);
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    MeshComponent->SetCollisionObjectType(ECC_WorldDynamic);
    
    Health = 100;
    MaxHealth = 100;
}

void ABlockActor::BeginPlay()
{
    Super::BeginPlay();
    MeshComponent->OnComponentHit.AddDynamic(this, &ABlockActor::OnHit);
}

void ABlockActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    
    if (GetActorLocation().Z < -500.0f)
    {
        Destroy();
    }
}

void ABlockActor::TakeDamage(int32 DamageAmount)
{
    Health -= DamageAmount;
    
    if (Health <= 0)
    {
        ABBGameMode* GameMode = Cast<ABBGameMode>(UGameplayStatics::GetGameMode(GetWorld()));
        if (GameMode)
        {
            GameMode->BlocksDestroyed++;
        }
        
        Destroy();
    }
}

void ABlockActor::SetHealth(int32 NewHealth)
{
    Health = NewHealth;
    MaxHealth = NewHealth;
}

void ABlockActor::OnHit(UPrimitiveComponent* HitComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit)
{
    AProjectile* Projectile = Cast<AProjectile>(OtherActor);
    if (Projectile)
    {
        TakeDamage(Projectile->Damage);
        Projectile->Destroy();
    }
    
    AFighterChar* Fighter = Cast<AFighterChar>(OtherActor);
    if (Fighter)
    {
        Fighter->TakeDamage(10.0f);
        TakeDamage(50);
    }
}
