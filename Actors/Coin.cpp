#include "Coin.h"
#include "Components/StaticMeshComponent.h"
#include "GameFramework/RotatingMovementComponent.h"
#include "FighterChar.h"
#include "BBGameState.h"

ACoin::ACoin()
{
    PrimaryActorTick.bCanEverTick = true;
    
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    RootComponent = MeshComponent;
    
    RotatingMovement = CreateDefaultSubobject<URotatingMovementComponent>(TEXT("RotatingMovement"));
    RotatingMovement->RotationRate = FRotator(0.0f, 180.0f, 0.0f);
    
    Value = 10;
    
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    MeshComponent->SetCollisionObjectType(ECC_WorldDynamic);
    MeshComponent->SetGenerateOverlapEvents(true);
}

void ACoin::BeginPlay()
{
    Super::BeginPlay();
    MeshComponent->OnComponentBeginOverlap.AddDynamic(this, &ACoin::OnCollect);
}

void ACoin::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void ACoin::OnCollect(UPrimitiveComponent* OverlappedComponent, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult)
{
    AFighterChar* Fighter = Cast<AFighterChar>(OtherActor);
    if (Fighter)
    {
        ABBGameState* GameState = Cast<ABBGameState>(GetWorld()->GetGameState());
        if (GameState)
        {
            GameState->AddScore(Value);
        }
        
        Destroy();
    }
}
