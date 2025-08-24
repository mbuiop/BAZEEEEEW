#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Coin.generated.h"

UCLASS()
class BLOCKBREAKER3D_API ACoin : public AActor
{
    GENERATED_BODY()
    
public:
    ACoin();
    
    virtual void Tick(float DeltaTime) override;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class UStaticMeshComponent* MeshComponent;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class URotatingMovementComponent* RotatingMovement;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Coin")
    int32 Value;

protected:
    virtual void BeginPlay() override;
    
    UFUNCTION()
    void OnCollect(UPrimitiveComponent* OverlappedComponent, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult);
};
