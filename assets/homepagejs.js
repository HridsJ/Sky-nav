
// Reactive particle background system with lighting effect
class ReactiveBackground {
    constructor() {
        this.canvas = document.getElementById('reactive-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: -999, y: -999 };
        this.particleCount = 150;
        this.maxDistance = 120;
        this.mouseRadius = 180;
        this.time = 0;
        
        this.init();
        this.animate();
        this.setupEventListeners();
    }

    init() {
        this.resizeCanvas();
        this.createParticles();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                baseOpacity: Math.random() * 0.015 + 0.05,
                currentOpacity: Math.random() * 0.3 + 0.1,
                size: Math.random() * 2 + 1,
                baseSize: Math.random() * 2 + 1,
                pulseSpeed: Math.random() * 0.02 + 0.01,
                pulseOffset: Math.random() * Math.PI * 2,
                driftX: (Math.random() - 0.5) * 0.5,
                driftY: (Math.random() - 0.5) * 0.5,
                originalX: 0,
                originalY: 0
            });
            
            // Store original position
            this.particles[i].originalX = this.particles[i].x;
            this.particles[i].originalY = this.particles[i].y;
        }
    }

    drawParticle(particle) {
        // Calculate glow intensity based on mouse proximity
        const dx = this.mouse.x - particle.x;
        const dy = this.mouse.y - particle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        let glowIntensity = 0;
        if (distance < this.mouseRadius) {
            glowIntensity = 1 - (distance / this.mouseRadius);
            glowIntensity = Math.pow(glowIntensity, 2); // Smoother falloff
        }

        // Base particle with subtle pulse
        const pulse = Math.sin(this.time * particle.pulseSpeed + particle.pulseOffset) * 0.2 + 0.6;
        const opacity = particle.baseOpacity * pulse + glowIntensity * 0.8;
        const size = particle.baseSize + glowIntensity * 2;

        // Outer glow effect
        if (glowIntensity > 0) {
            const glowSize = size * (3 + glowIntensity * 2);
            const gradient = this.ctx.createRadialGradient(
                particle.x, particle.y, 0,
                particle.x, particle.y, glowSize
            );
            gradient.addColorStop(0, `rgba(59, 130, 246, ${glowIntensity * 0.01})`);
            gradient.addColorStop(0.3, `rgba(139, 92, 246, ${glowIntensity * 0.005})`);
            gradient.addColorStop(0.6, `rgba(6, 182, 212, ${glowIntensity * 0.0003})`);
            gradient.addColorStop(1, `rgba(59, 130, 246, 0)`);
            
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, glowSize, 0, Math.PI * 2);
            this.ctx.fillStyle = gradient;
            this.ctx.fill();
        }

        // Core particle
        this.ctx.beginPath();
        this.ctx.arc(particle.x, particle.y, size, 0, Math.PI * 2);
        this.ctx.fillStyle = `rgba(59, 130, 246, ${Math.min(1, opacity)})`;
        this.ctx.fill();

        // Bright center for lit particles
        if (glowIntensity > 0.3) {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, size * 0.5, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(255, 255, 255, ${glowIntensity * 0.3})`;
            this.ctx.fill();
        }
    }

    drawConnection(p1, p2, distance) {
        // Calculate lighting effect for connections
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const midDx = this.mouse.x - midX;
        const midDy = this.mouse.y - midY;
        const midDistance = Math.sqrt(midDx * midDx + midDy * midDy);
        
        let connectionGlow = 0;
        if (midDistance < this.mouseRadius) {
            connectionGlow = 1 - (midDistance / this.mouseRadius);
            connectionGlow = Math.pow(connectionGlow, 2);
        }

        const baseOpacity = Math.pow(1 - (distance / this.maxDistance), 2) * 0.01;
        const finalOpacity = baseOpacity + connectionGlow * 0.7;
        
        if (finalOpacity > 0.05) {
            // Enhanced connection with gradient
            const gradient = this.ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
            gradient.addColorStop(0, `rgba(59, 130, 246, ${finalOpacity})`);
            gradient.addColorStop(0.5, `rgba(139, 92, 246, ${finalOpacity * 0.8})`);
            gradient.addColorStop(1, `rgba(59, 130, 246, ${finalOpacity})`);
            
            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = Math.max(0.3, finalOpacity * 0.2);
            this.ctx.stroke();
        }
    }

    updateParticle(particle) {
        // Gentle drift animation - particles stay in place but drift slightly
        particle.x = particle.originalX + Math.sin(this.time * particle.pulseSpeed) * 20 + particle.driftX * this.time * 0.1;
        particle.y = particle.originalY + Math.cos(this.time * particle.pulseSpeed * 0.8) * 15 + particle.driftY * this.time * 0.1;

        // Wrap around screen edges
        if (particle.x < -50) {
            particle.originalX = this.canvas.width + 50;
            particle.x = particle.originalX;
        }
        if (particle.x > this.canvas.width + 50) {
            particle.originalX = -50;
            particle.x = particle.originalX;
        }
        if (particle.y < -50) {
            particle.originalY = this.canvas.height + 50;
            particle.y = particle.originalY;
        }
        if (particle.y > this.canvas.height + 50) {
            particle.originalY = -50;
            particle.y = particle.originalY;
        }
    }

    animate() {
        this.time += 0.016; // Roughly 60fps timing
        
        // Clear canvas with fade effect
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Update particles
        this.particles.forEach(particle => {
            this.updateParticle(particle);
        });

        // Draw connections first (behind particles)
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.maxDistance) {
                    this.drawConnection(this.particles[i], this.particles[j], distance);
                }
            }
        }

        // Draw particles on top
        this.particles.forEach(particle => {
            this.drawParticle(particle);
        });

        requestAnimationFrame(() => this.animate());
    }

    setupEventListeners() {
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });

        window.addEventListener('mouseleave', () => {
            this.mouse.x = -999;
            this.mouse.y = -999;
        });

        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.createParticles();
        });

        // Touch support for mobile
        window.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches[0]) {
                this.mouse.x = e.touches[0].clientX;
                this.mouse.y = e.touches[0].clientY;
            }
        }, { passive: false });

        window.addEventListener('touchstart', (e) => {
            if (e.touches[0]) {
                this.mouse.x = e.touches[0].clientX;
                this.mouse.y = e.touches[0].clientY;
            }
        });

        window.addEventListener('touchend', () => {
            this.mouse.x = -999;
            this.mouse.y = -999;
        });
    }
}

let hasNavigated = false;

// Apple-style scroll animation
function handleScroll() {
    if (hasNavigated) return;

    const scrollTop = window.pageYOffset;
    const windowHeight = window.innerHeight;
    const launchSection = document.querySelector('.launch-section');
    const launchText = document.getElementById('launchText');
    const previewContainer = document.getElementById('previewContainer');
    
    if (!launchSection) return;

    const sectionTop = launchSection.offsetTop;
    const sectionHeight = launchSection.offsetHeight;
    const progress = Math.max(0, Math.min(1, (scrollTop - sectionTop) / (sectionHeight - windowHeight)));

    if (progress > 0.1) {
        launchText.classList.add('visible');
    }

    if (progress > 0.3) {
        const scaleProgress = Math.min(1, (progress - 0.3) / 0.4);
        const scale = 1 + (scaleProgress * 2);
        const brightness = 1 + (scaleProgress * 0.3);
        
        previewContainer.style.transform = `scale(${scale})`;
        previewContainer.style.filter = `brightness(${brightness})`;
        previewContainer.style.boxShadow = `0 ${20 + scaleProgress * 40}px ${60 + scaleProgress * 100}px rgba(59, 130, 246, ${0.3 + scaleProgress * 0.4})`;

        if (scaleProgress > 0.85 && !hasNavigated) {
            hasNavigated = true;
            triggerTransition();
        }
    }
}

function triggerTransition() {
    const overlay = document.getElementById('transitionOverlay');
    const transitionPreview = document.getElementById('transitionPreview');
    
    // Show overlay
    overlay.classList.add('active');
    
    // Expand the preview to fullscreen
    setTimeout(() => {
        transitionPreview.classList.add('expand');
    }, 100);
    
    // Navigate after animation completes
    setTimeout(() => {
        window.location.href = './Location_page.html';
    }, 1500);
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize reactive background
    new ReactiveBackground();
    
    // Add hover effects to link cards
    const linkCards = document.querySelectorAll('.link-card');
    linkCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            if (!this.querySelector('.link-button[style*="cursor: default"]')) {
                this.style.transform = 'translateY(-8px) scale(1.02)';
            }
        });
        card.addEventListener('mouseleave', function() {
            if (!this.querySelector('.link-button[style*="cursor: default"]')) {
                this.style.transform = 'translateY(0) scale(1)';
            }
        });
    });

    // Click handler for preview container
    document.getElementById('previewContainer').addEventListener('click', () => {
        if (!hasNavigated) {
            hasNavigated = true;
            triggerTransition();
        }
    });

    window.addEventListener('scroll', handleScroll);
});

// Parallax effect for hero section
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const rate = scrolled * -0.5;
    hero.style.transform = `translateY(${rate}px)`;
});