#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "FighterChar.generated.h"

UCLASS()
class BLOCKBREAKER3D_API AFighterChar : public ACharacter
{
    GENERATED_BODY()

public:
    AFighterChar();
    
    virtual void Tick(float DeltaTime) override;
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;
    
    UFUNCTION(BlueprintCallable)
    void FireWeapon();
    
    UFUNCTION(BlueprintCallable)
    void TakeDamage(float DamageAmount);
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class UHealthComp* HealthComponent;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class UWeaponComp* WeaponComponent;

protected:
    virtual void BeginPlay() override;
    
    void MoveHorizontal(float Value);
    void MoveVertical(float Value);
};
