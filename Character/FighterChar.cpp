#include "FighterChar.h"
#include "Components/HealthComp.h"
#include "Components/WeaponComp.h"
#include "Components/InputComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"

AFighterChar::AFighterChar()
{
    PrimaryActorTick.bCanEverTick = true;
    
    HealthComponent = CreateDefaultSubobject<UHealthComp>(TEXT("HealthComponent"));
    WeaponComponent = CreateDefaultSubobject<UWeaponComp>(TEXT("WeaponComponent"));
    
    GetCharacterMovement()->bOrientRotationToMovement = true;
    GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f);
    GetCharacterMovement()->AirControl = 0.2f;
}

void AFighterChar::BeginPlay()
{
    Super::BeginPlay();
}

void AFighterChar::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void AFighterChar::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);
    
    PlayerInputComponent->BindAxis("MoveHorizontal", this, &AFighterChar::MoveHorizontal);
    PlayerInputComponent->BindAxis("MoveVertical", this, &AFighterChar::MoveVertical);
    PlayerInputComponent->BindAction("Fire", IE_Pressed, this, &AFighterChar::FireWeapon);
}

void AFighterChar::MoveHorizontal(float Value)
{
    if (Value != 0.0f && Controller)
    {
        const FRotator Rotation = Controller->GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
        AddMovementInput(Direction, Value);
    }
}

void AFighterChar::MoveVertical(float Value)
{
    if (Value != 0.0f && Controller)
    {
        const FRotator Rotation = Controller->GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }
}

void AFighterChar::FireWeapon()
{
    if (WeaponComponent)
    {
        WeaponComponent->Fire();
    }
}

void AFighterChar::TakeDamage(float DamageAmount)
{
    if (HealthComponent)
    {
        HealthComponent->TakeDamage(DamageAmount);
    }
}
