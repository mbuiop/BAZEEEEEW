#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BlockActor.generated.h"

UCLASS()
class BLOCKBREAKER3D_API ABlockActor : public AActor
{
    GENERATED_BODY()
    
public:
    ABlockActor();
    
    virtual void Tick(float DeltaTime) override;
    
    UFUNCTION(BlueprintCallable)
    void TakeDamage(int32 DamageAmount);
    
    UFUNCTION(BlueprintCallable)
    void SetHealth(int32 NewHealth);
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Block")
    int32 Health;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Block")
    int32 MaxHealth;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class UStaticMeshComponent* MeshComponent;

protected:
    virtual void BeginPlay() override;
    
    UFUNCTION()
    void OnHit(UPrimitiveComponent* HitComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit);
};
