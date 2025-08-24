#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "WeaponComp.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class BLOCKBREAKER3D_API UWeaponComp : public UActorComponent
{
    GENERATED_BODY()

public:
    UWeaponComp();
    
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
    
    UFUNCTION(BlueprintCallable)
    void Fire();
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
    TSubclassOf<class AProjectile> ProjectileClass;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
    float FireRate;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
    FVector MuzzleOffset;

protected:
    virtual void BeginPlay() override;
    
private:
    FTimerHandle TimerHandle_ShotTimerExpired;
    bool bCanFire;
    
    void ResetShot();
};
